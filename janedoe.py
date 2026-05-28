import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# DETECÇÃO DE CRISTAIS URINÁRIOS
# - CISTINA (hexagonal)
# - BILIRRUBINA (cor amarela/marrom)
#
# AGORA COM FILTRO DE TAMANHO
# =========================================================

# =========================
# CARREGAR IMAGEM
# =========================

imagem = cv2.imread("Teste3.png")

rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

# =========================
# PRÉ-PROCESSAMENTO
# =========================

blur = cv2.GaussianBlur(gray, (5,5), 0)

bordas = cv2.Canny(blur, 40, 120)

kernel = np.ones((3,3), np.uint8)

fechado = cv2.morphologyEx(
    bordas,
    cv2.MORPH_CLOSE,
    kernel
)

resultado = rgb.copy()

# =========================================================
# FILTRO DE TAMANHO
# =========================================================

# Área mínima:
# remove cristais muito pequenos
AREA_MINIMA = 80

# Área máxima:
# evita manchas gigantes
AREA_MAXIMA = 8000

# =========================================================
# 1. DETECÇÃO DE CISTINA (HEXAGONAL)
# =========================================================

contornos, _ = cv2.findContours(
    fechado,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

hexagonos = 0

for cnt in contornos:

    area = cv2.contourArea(cnt)

    # =========================
    # FILTRO DE TAMANHO
    # =========================
    if area < AREA_MINIMA or area > AREA_MAXIMA:
        continue

    perimetro = cv2.arcLength(cnt, True)

    aprox = cv2.approxPolyDP(
        cnt,
        0.03 * perimetro,
        True
    )

    lados = len(aprox)

    # Aproximadamente hexagonal
    if 5 <= lados <= 7:

        circularidade = (
            4 * np.pi * area
        ) / (perimetro * perimetro + 1e-5)

        if 0.4 < circularidade < 0.95:

            hexagonos += 1

            cv2.drawContours(
                resultado,
                [aprox],
                -1,
                (0,255,0),
                2
            )

            x, y, w, h = cv2.boundingRect(aprox)

            cv2.putText(
                resultado,
                f"Cistina ({int(area)})",
                (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0,255,0),
                1
            )

# =========================================================
# 2. DETECÇÃO DE BILIRRUBINA (COR)
# =========================================================

# Tons amarelos/marrons
lower = np.array([10, 60, 40])
upper = np.array([40, 255, 255])

mascara_bili = cv2.inRange(
    hsv,
    lower,
    upper
)

mascara_bili = cv2.morphologyEx(
    mascara_bili,
    cv2.MORPH_OPEN,
    kernel
)

contornos_bili, _ = cv2.findContours(
    mascara_bili,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

bilirrubinas = 0

for cnt in contornos_bili:

    area = cv2.contourArea(cnt)

    # =========================
    # FILTRO DE TAMANHO
    # =========================
    if area < AREA_MINIMA or area > AREA_MAXIMA:
        continue

    bilirrubinas += 1

    x, y, w, h = cv2.boundingRect(cnt)

    cv2.rectangle(
        resultado,
        (x, y),
        (x+w, y+h),
        (255,0,0),
        2
    )

    cv2.putText(
        resultado,
        f"Bilirrubina ({int(area)})",
        (x, y-5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255,0,0),
        1
    )

# =========================================================
# RESULTADOS
# =========================================================

print("="*40)
print(f"Cristais de Cistina: {hexagonos}")
print(f"Cristais de Bilirrubina: {bilirrubinas}")
print("="*40)

# =========================================================
# VISUALIZAÇÃO
# =========================================================

plt.figure(figsize=(18,6))

plt.subplot(1,4,1)
plt.imshow(rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(fechado, cmap="gray")
plt.title("Bordas")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(mascara_bili, cmap="gray")
plt.title("Mascara Bilirrubina")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(resultado)
plt.title("Resultado Final")
plt.axis("off")

plt.tight_layout()
plt.show()