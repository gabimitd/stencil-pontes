# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import hashlib
import xml.etree.ElementTree as ET

try:
    import cairosvg
    HAS_CAIRO = True
except Exception:
    HAS_CAIRO = False

try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    HAS_SVGLIB = True
except Exception:
    HAS_SVGLIB = False

HAS_SVG = HAS_CAIRO or HAS_SVGLIB

# ── Página ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gerador de Stencil", layout="wide")
st.title("Gerador de Stencil com Pontes")

if not HAS_SVG:
    st.info("Suporte a SVG indisponivel. Instale: `python -m pip install cairosvg`")
elif HAS_SVGLIB and not HAS_CAIRO:
    st.warning(
        "SVG ativo via svglib (modo compativel Windows). "
        "**Se o SVG contem texto**, converta-o em caminhos antes de exportar "
        "(Inkscape: Caminho > Objeto para Caminho | Illustrator: Texto > Criar Contornos), "
        "caso contrario o texto pode aparecer com buracos."
    )

# ── Controles ─────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)
with col_a:
    thickness = st.slider("Espessura da ponte (px)", 2, 40, 10)
    min_area  = st.slider("Area minima do buraco (px²)", 50, 5000, 300,
                          help="Buracos menores sao ignorados (evita pontes em ruidos)")
with col_b:
    max_bridges = st.slider("Pontes por buraco", 1, 4, 1,
                            help="Use 2-4 para buracos grandes ou irregulares")
    if HAS_SVG and HAS_CAIRO:
        svg_dpi = st.slider("DPI (qualidade SVG)", 72, 300, 150,
                            help="Resolucao usada ao rasterizar arquivos SVG (cairosvg)")
    else:
        svg_dpi = 150

export_svg = st.checkbox("Exportar resultado como SVG vetorial", value=False)

# ── Upload ────────────────────────────────────────────────────────────────────
file_types = ["png", "jpg", "jpeg"]
if HAS_SVG:
    file_types = ["svg"] + file_types

uploaded_file = st.file_uploader(
    "Envie uma imagem (PNG/JPG/JPEG) ou arquivo SVG",
    type=file_types,
)

# ══════════════════════════════════════════════════════════════════════════════
# Funcoes auxiliares — pontes
# ══════════════════════════════════════════════════════════════════════════════

def touches_border(contour, img_shape):
    """Retorna True se o contorno toca as bordas da imagem."""
    h, w = img_shape[:2]
    for p in contour:
        x, y = p[0]
        if x <= 1 or y <= 1 or x >= w - 2 or y >= h - 2:
            return True
    return False


def _subsample(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) > n:
        idx = np.linspace(0, len(arr) - 1, n, dtype=int)
        return arr[idx]
    return arr


def compute_outward_normals(pts: np.ndarray) -> np.ndarray:
    """
    Vetor unitario do centroide ao ponto — aproxima o normal outward
    de cada vertice do contorno de um buraco.
    """
    centroid = pts.mean(axis=0)
    vecs = pts - centroid
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return vecs / norms


def compute_tangent_normals_oriented(pts: np.ndarray, toward: np.ndarray) -> np.ndarray:
    """
    Normais geometricas (perpendiculares ao tangente) de cada ponto do contorno,
    orientadas para apontar NA DIRECAO de 'toward'.
    Mais preciso que a abordagem centroidal para contornos complexos.
    """
    prev_pts = np.roll(pts, 1, axis=0)
    next_pts = np.roll(pts, -1, axis=0)
    tangents = (next_pts - prev_pts).astype(float)
    # Rotacionar 90 graus para obter normal
    normals = np.stack([tangents[:, 1], -tangents[:, 0]], axis=1)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    normals = normals / norms
    # Garantir que aponta em direcao a 'toward'
    to_target = toward[None, :] - pts
    dots = (normals * to_target).sum(axis=1)
    normals[dots < 0] *= -1
    return normals


def _score_matrix(inner_pts, inner_normals, outer_pts, outer_normals):
    """
    Calcula a matriz de scores (N, M) para todos os pares inner x outer.
    Score = (align_inner * align_outer) / distancia.
    Pares invalidos recebem -inf.
    Retorna (scores, ip, op, dists).
    """
    MAX = 400
    ip = _subsample(inner_pts, MAX).astype(float)
    iN = _subsample(inner_normals, MAX).astype(float)
    op = _subsample(outer_pts, MAX).astype(float)
    oN = _subsample(outer_normals, MAX).astype(float)

    diff   = op[None, :, :] - ip[:, None, :]
    dists  = np.sqrt((diff ** 2).sum(axis=2))
    safe_d = np.where(dists < 1e-9, 1e-9, dists)
    dirs   = diff / safe_d[:, :, None]

    align_i = (dirs * iN[:, None, :]).sum(axis=2)
    align_o = (dirs * oN[None, :, :]).sum(axis=2)

    valid  = (align_i > 0.15) & (align_o > 0.15)
    scores = np.where(valid, (align_i * align_o) / safe_d, -np.inf)

    # fallback: so lado interno
    if scores.max() == -np.inf:
        scores = np.where(align_i > 0.0, align_i ** 2 / safe_d, -np.inf)
    # fallback: distancia pura (negativa para que argmax escolha o mais proximo)
    if scores.max() == -np.inf:
        scores = -dists

    return scores, ip, op, dists


def find_top_k_bridges(
    inner_pts: np.ndarray,
    inner_normals: np.ndarray,
    outer_pts: np.ndarray,
    outer_normals: np.ndarray,
    k: int = 6,
) -> list:
    """
    Retorna os top-k candidatos de ponte [(pt_inner, pt_outer), ...],
    ordenados do melhor para o pior score.
    Garante diversidade: descarta candidatos muito proximos a um ja selecionado.
    """
    scores, ip, op, dists = _score_matrix(
        inner_pts, inner_normals, outer_pts, outer_normals
    )

    # min_sep: distancia minima entre pontos de ancoragem de candidatos diferentes
    min_sep = max(5.0, float(np.sqrt(dists.min()) * 0.5))

    flat_order = np.argsort(scores.ravel())[::-1]
    results = []

    for flat_idx in flat_order:
        if len(results) >= k:
            break
        i_idx, o_idx = np.unravel_index(flat_idx, scores.shape)
        pt_i = tuple(int(v) for v in ip[i_idx])
        pt_o = tuple(int(v) for v in op[o_idx])

        # Checar diversidade: nao adicionar se muito proximo a candidato anterior
        too_close = any(
            np.hypot(pt_i[0] - prev_i[0], pt_i[1] - prev_i[1]) < min_sep
            for prev_i, _ in results
        )
        if not too_close:
            results.append((pt_i, pt_o))

    return results if results else [
        (tuple(int(v) for v in ip[0]), tuple(int(v) for v in op[0]))
    ]


def find_bridge_candidates_for_hole(
    inner_contour: np.ndarray,
    outer_contour: np.ndarray,
    n_bridges: int = 1,
    k: int = 6,
) -> list:
    """
    Calcula candidatos de pontes para um buraco.
    Retorna lista de n_bridges listas de candidatos:
        [ [(pt_i, pt_o), ...], [(pt_i, pt_o), ...], ... ]
    """
    inner_pts = inner_contour[:, 0, :].astype(float)
    outer_pts = outer_contour[:, 0, :].astype(float)

    inner_centroid = inner_pts.mean(axis=0)
    inner_normals  = compute_outward_normals(inner_pts)
    outer_normals  = -compute_tangent_normals_oriented(outer_pts, inner_centroid)

    outer_sub  = _subsample(outer_pts, 400)
    outer_nsub = _subsample(outer_normals, 400)

    N          = len(inner_pts)
    all_cands  = []

    for b in range(n_bridges):
        start = int(b * N / n_bridges)
        end   = int((b + 1) * N / n_bridges)
        if end <= start:
            continue
        arc_pts = _subsample(inner_pts[start:end], 150)
        arc_nrm = _subsample(inner_normals[start:end], 150)
        cands   = find_top_k_bridges(arc_pts, arc_nrm, outer_sub, outer_nsub, k=k)
        all_cands.append(cands)

    return all_cands


def find_bridge_positions_for_hole(inner_contour, outer_contour, n_bridges=1):
    """Conveniencia: retorna apenas o melhor candidato por arco."""
    return [cands[0] for cands in
            find_bridge_candidates_for_hole(inner_contour, outer_contour, n_bridges, k=1)]


def draw_bridges_for_hole(
    img: np.ndarray,
    inner_contour: np.ndarray,
    outer_contour: np.ndarray,
    thickness: int,
    n_bridges: int = 1,
) -> int:
    """Calcula e desenha n_bridges pontes com espessura uniforme."""
    positions = find_bridge_positions_for_hole(inner_contour, outer_contour, n_bridges)
    for pt_i, pt_o in positions:
        cv2.line(img, pt_i, pt_o, 255, thickness)
    return len(positions)


# ══════════════════════════════════════════════════════════════════════════════
# Funcoes SVG
# ══════════════════════════════════════════════════════════════════════════════

def _parse_svg_meta(file_bytes: bytes) -> dict:
    """Extrai viewBox e dimensoes do SVG via ElementTree."""
    meta = {"vb_x": 0.0, "vb_y": 0.0, "vb_w": None, "vb_h": None,
            "orig_width": None, "orig_height": None}
    try:
        root = ET.fromstring(file_bytes)
        meta["orig_width"]  = root.get("width", "")
        meta["orig_height"] = root.get("height", "")
        vb = root.get("viewBox", "").strip().replace(",", " ")
        if vb:
            parts = [float(x) for x in vb.split()]
            if len(parts) == 4:
                meta["vb_x"], meta["vb_y"] = parts[0], parts[1]
                meta["vb_w"], meta["vb_h"] = parts[2], parts[3]
    except ET.ParseError:
        pass
    return meta


def _pil_to_gray_meta(pil_img: Image.Image, meta: dict) -> tuple:
    """Converte imagem PIL RGBA para grayscale e completa o meta dict."""
    bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
    bg.paste(pil_img, mask=pil_img.split()[3])
    gray = bg.convert("L")
    px_w, px_h = gray.size
    meta["px_w"] = px_w
    meta["px_h"] = px_h
    if meta["vb_w"] is None:
        meta["vb_w"] = float(px_w)
        meta["vb_h"] = float(px_h)
    return np.array(gray), meta


def svg_to_gray(file_bytes: bytes, dpi: int) -> tuple:
    """
    Rasteriza SVG para grayscale numpy array.
    Usa cairosvg se disponivel, senao svglib.
    Retorna (gray_array, meta_dict).
    """
    meta = _parse_svg_meta(file_bytes)

    if HAS_CAIRO:
        png_data = cairosvg.svg2png(bytestring=file_bytes, dpi=dpi)
        rgba = Image.open(io.BytesIO(png_data)).convert("RGBA")
        return _pil_to_gray_meta(rgba, meta)

    # fallback: svglib
    rlg = svg2rlg(io.BytesIO(file_bytes))
    if rlg is None:
        raise ValueError("svglib nao conseguiu ler o arquivo SVG.")
    pil_img = renderPM.drawToPIL(rlg, dpi=dpi, fmt="PNG")
    rgba = pil_img.convert("RGBA")
    return _pil_to_gray_meta(rgba, meta)


def _contour_to_path_d(contour: np.ndarray, sx: float, sy: float) -> str:
    """Converte contorno OpenCV em string de path SVG (M … L … Z)."""
    epsilon = max(0.5, contour.shape[0] / 1000.0)
    approx  = cv2.approxPolyDP(contour, epsilon, closed=True)
    pts     = approx[:, 0, :]
    if len(pts) < 2:
        return ""
    parts = [f"M {pts[0][0] * sx:.3f} {pts[0][1] * sy:.3f}"]
    for pt in pts[1:]:
        parts.append(f"L {pt[0] * sx:.3f} {pt[1] * sy:.3f}")
    parts.append("Z")
    return " ".join(parts)


def result_to_svg_bytes(result_img: np.ndarray, meta: dict) -> bytes:
    """
    Converte imagem binaria processada de volta para SVG vetorial.
    Areas pretas (recortes) viram paths com fill preto.
    fill-rule=evenodd faz buracos filhos funcionarem corretamente.
    """
    px_w = meta["px_w"]
    px_h = meta["px_h"]
    vb_w = meta["vb_w"]
    vb_h = meta["vb_h"]
    vb_x = meta.get("vb_x", 0.0)
    vb_y = meta.get("vb_y", 0.0)

    sx = vb_w / px_w
    sy = vb_h / px_h

    w_attr = meta.get("orig_width")  or f"{vb_w:.3f}"
    h_attr = meta.get("orig_height") or f"{vb_h:.3f}"

    # regioes pretas = recortes → inverter para findContours achar regioes brancas
    inverted = cv2.bitwise_not(result_img)
    contours, hierarchy = cv2.findContours(
        inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS
    )

    path_elements = []
    if hierarchy is not None:
        hier = hierarchy[0]
        for i, h in enumerate(hier):
            if h[3] != -1:
                continue  # pular filhos (sao adicionados como sub-paths do pai)

            d = _contour_to_path_d(contours[i], sx, sy)
            if not d:
                continue

            # adicionar filhos como sub-paths (fill-rule evenodd cria buracos)
            child = h[2]
            while child != -1:
                child_d = _contour_to_path_d(contours[child], sx, sy)
                if child_d:
                    d += " " + child_d
                child = hier[child][0]  # proximo irmao

            path_elements.append(f'  <path d="{d}" fill="black" fill-rule="evenodd"/>')

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' width="{w_attr}" height="{h_attr}"'
        f' viewBox="{vb_x} {vb_y} {vb_w:.3f} {vb_h:.3f}">',
        f'  <rect width="{vb_w:.3f}" height="{vb_h:.3f}" fill="white"/>',
    ] + path_elements + ["</svg>"]

    return "\n".join(lines).encode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _compute_binary_and_contours(file_bytes: bytes, is_svg: bool, dpi: int):
    """Cache da binarizacao e deteccao de contornos — evita reprocessar ao mover sliders."""
    if is_svg:
        img, meta = svg_to_gray(file_bytes, dpi)
    else:
        pil_raw = Image.open(io.BytesIO(file_bytes))
        # Compositar sobre fundo branco antes de converter para L
        # (resolve PNGs com canal alpha / fundo transparente)
        if pil_raw.mode in ("RGBA", "LA", "PA"):
            bg = Image.new("RGBA", pil_raw.size, (255, 255, 255, 255))
            bg.paste(pil_raw.convert("RGBA"), mask=pil_raw.split()[-1])
            pil = bg.convert("L")
        else:
            pil = pil_raw.convert("L")
        img = np.array(pil)
        h, w = img.shape
        meta = {
            "px_w": w, "px_h": h,
            "vb_w": float(w), "vb_h": float(h),
            "vb_x": 0.0,  "vb_y": 0.0,
            "orig_width": None, "orig_height": None,
        }

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    inverted = cv2.bitwise_not(binary)
    contours, hierarchy = cv2.findContours(
        inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    return binary, contours, hierarchy, meta


if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    is_svg     = HAS_SVG and uploaded_file.name.lower().endswith(".svg")

    # ── Carregar + binarizar (com cache) ──────────────────────────────────────
    with st.spinner("Processando imagem..."):
        try:
            binary, contours, hierarchy, meta = _compute_binary_and_contours(
                file_bytes, is_svg, svg_dpi
            )
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")
            st.stop()

    # ── Calcular candidatos de todas as pontes ────────────────────────────────
    # all_candidates: lista de listas — para cada ponte, top-K posicoes candidatas
    all_candidates = []
    skipped_count  = 0

    if hierarchy is not None:
        hier = hierarchy[0]
        for i, h in enumerate(hier):
            parent = h[3]
            if parent == -1:
                continue
            area = cv2.contourArea(contours[i])
            if area < min_area:
                skipped_count += 1
                continue
            if touches_border(contours[i], binary.shape):
                skipped_count += 1
                continue
            hole_cands = find_bridge_candidates_for_hole(
                contours[i], contours[parent], max_bridges, k=6
            )
            all_candidates.extend(hole_cands)

    n_bridges_total = len(all_candidates)

    # ── Session state: hash muda quando imagem ou parametros de deteccao mudam ─
    _hash      = hashlib.md5(
        file_bytes + f"{max_bridges}_{min_area}".encode()
    ).hexdigest()[:10]
    _key_thick = f"bth_{_hash}"
    _key_sel   = f"sel_{_hash}"

    # Inicializar ou resetar se numero de pontes mudou
    if (_key_thick not in st.session_state
            or len(st.session_state[_key_thick]) != n_bridges_total):
        st.session_state[_key_thick] = [thickness] * n_bridges_total
        st.session_state[_key_sel]   = [0] * n_bridges_total

    bridge_thicknesses = st.session_state[_key_thick]
    selections         = st.session_state[_key_sel]

    # ── Metricas ───────────────────────────────────────────────────────────────
    m1, m2 = st.columns(2)
    m1.metric("Pontes criadas", n_bridges_total)
    m2.metric("Buracos ignorados", skipped_count)

    # ── Ajuste individual: espessura + botao regenerar ────────────────────────
    if all_candidates:
        with st.expander(f"Ajuste individual ({n_bridges_total} pontes)"):
            if st.button("Resetar todas ao valor global"):
                st.session_state[_key_thick] = [thickness] * n_bridges_total
                st.session_state[_key_sel]   = [0] * n_bridges_total
                bridge_thicknesses = st.session_state[_key_thick]
                selections         = st.session_state[_key_sel]

            n_cols = min(4, n_bridges_total)
            grid   = st.columns(n_cols)

            for idx, cands in enumerate(all_candidates):
                sel = selections[idx] % len(cands)
                with grid[idx % n_cols]:
                    # Botao regenerar: cicla para o proximo candidato
                    label_regen = f"↻  Ponte {idx+1}  ({sel+1}/{len(cands)})"
                    if st.button(label_regen, key=f"regen_{_hash}_{idx}"):
                        selections[idx] = (sel + 1) % len(cands)
                        st.session_state[_key_sel] = selections
                        st.rerun()
                    # Controle de espessura
                    bridge_thicknesses[idx] = st.number_input(
                        "Espessura",
                        min_value=1, max_value=60,
                        value=bridge_thicknesses[idx],
                        key=f"bt_{_hash}_{idx}",
                        label_visibility="collapsed",
                    )

            st.session_state[_key_thick] = bridge_thicknesses

    # ── Desenhar usando o candidato selecionado para cada ponte ───────────────
    result = binary.copy()
    active_bridges = []   # posicoes efetivamente desenhadas (para anotacao)
    for idx, cands in enumerate(all_candidates):
        sel   = selections[idx] % len(cands)
        pt_i, pt_o = cands[sel]
        t     = bridge_thicknesses[idx]
        cv2.line(result, pt_i, pt_o, 255, t)
        active_bridges.append((pt_i, pt_o))

    # ── Preview anotado: numera cada ponte na imagem ──────────────────────────
    h_img, w_img = result.shape
    font_scale   = max(0.35, min(w_img, h_img) / 600)
    font_thick   = max(1, round(font_scale * 2))
    result_rgb   = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    for idx, (pt_i, pt_o) in enumerate(active_bridges):
        mid = ((pt_i[0] + pt_o[0]) // 2, (pt_i[1] + pt_o[1]) // 2)
        cv2.putText(result_rgb, str(idx + 1), mid,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 30, 30), font_thick,
                    cv2.LINE_AA)

    # atualizar dimensoes do meta
    meta["px_w"] = result.shape[1]
    meta["px_h"] = result.shape[0]

    # ── Preview ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.image(binary, caption="Binario original", use_container_width=True)
    with col2:
        st.image(result_rgb, caption="Resultado com pontes (numeros = ID da ponte)",
                 use_container_width=True)

    # ── Downloads (sem anotacoes) ──────────────────────────────────────────────
    buf = io.BytesIO()
    Image.fromarray(result).save(buf, format="PNG")

    d1, d2 = st.columns(2)
    d1.download_button(
        label="Baixar PNG",
        data=buf.getvalue(),
        file_name="stencil_corrigido.png",
        mime="image/png",
    )

    if export_svg:
        svg_data = result_to_svg_bytes(result, meta)
        d2.download_button(
            label="Baixar SVG vetorial",
            data=svg_data,
            file_name="stencil_corrigido.svg",
            mime="image/svg+xml",
        )
