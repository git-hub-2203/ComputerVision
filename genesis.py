import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Imagens de teste
imagens = [
    "Teste1.png",
    "Teste3.png",
    "Teste4.png",
    "Teste5.png",
    "Teste6.png",
    "Teste7.png",
    "Teste8.png",
]

def detectar_cristais(caminho):
    img = cv2.imread(caminho)
    original = img.copy()

    # 1. Cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Melhorar contraste local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 3. Suavizar ruído
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Segmentação por borda
    edges = cv2.Canny(blur, 25, 90)

    # 5. Fechar falhas nas bordas
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # 6. Encontrar contornos
    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detectados = []

    for c in contornos:
        area = cv2.contourArea(c)
        if area < 25 or area > 8000:
            continue

        perimetro = cv2.arcLength(c, True)
        if perimetro == 0:
            continue

        x, y, w, h = cv2.boundingRect(c)
        razao = w / float(h)

        circularidade = 4 * np.pi * area / (perimetro ** 2)

        # Aproximação poligonal: útil para cistina/hexágonos
        approx = cv2.approxPolyDP(c, 0.04 * perimetro, True)
        vertices = len(approx)

        # Filtros gerais:
        # - cristais hexagonais: 5 a 7 vértices
        # - cristais aglomerados/radiais: contorno irregular, circularidade média
        possivel_hexagono = 5 <= vertices <= 7 and 0.55 <= razao <= 1.45
        possivel_aglomerado = area > 60 and 0.25 <= circularidade <= 0.95

        if possivel_hexagono or possivel_aglomerado:
            detectados.append(c)

            if possivel_hexagono:
                cor = (255, 0, 0)   # azul: possível cistina
                label = "hexagonal"
            else:
                cor = (0, 255, 0)   # verde: possível cristal/agregado
                label = "cristal"

            cv2.drawContours(original, [c], -1, cor, 2)
            cv2.putText(
                original,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                cor,
                1
            )

    return img, gray, edges, original, len(detectados)


for caminho in imagens:
    img, gray, edges, result, qtd = detectar_cristais(caminho)

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap="gray")
    plt.title("Bordas - Canny")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f"Detectados: {qtd}")
    plt.axis("off")

    plt.suptitle(caminho)
    plt.tight_layout()
    plt.show()