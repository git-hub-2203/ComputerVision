import cv2
import numpy as np

img = cv2.imread("ImagemTeste5.png")
output = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# melhora contraste
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)

# reduz ruído
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# detecta bordas dos cristais
edges = cv2.Canny(blur, 35, 120)

# fecha pequenas falhas nas bordas
kernel = np.ones((3, 3), np.uint8)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

# encontra contornos
contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # filtra ruídos pequenos e objetos grandes demais
    if w < 10 or h < 10:
        continue
    if w > 120 or h > 120:
        continue

    # desenha apenas o contorno
    cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)

cv2.imshow("Contorno dos cristais", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("cristais_contornados.png", output)