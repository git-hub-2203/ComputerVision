const imageInput = document.getElementById("imageInput");
const analyzeButton = document.getElementById("analyzeButton");
const statusPill = document.getElementById("statusPill");
const objectsCount = document.getElementById("objectsCount");
const coverageValue = document.getElementById("coverageValue");
const intensityValue = document.getElementById("intensityValue");
const colorValue = document.getElementById("colorValue");
const detectedList = document.getElementById("detectedList");
const analysisSummary = document.getElementById("analysisSummary");

const originalCanvas = document.getElementById("originalCanvas");
const processedCanvas = document.getElementById("processedCanvas");
const edgesCanvas = document.getElementById("edgesCanvas");
const overlayCanvas = document.getElementById("overlayCanvas");

const originalCtx = originalCanvas.getContext("2d", { willReadFrequently: true });
const processedCtx = processedCanvas.getContext("2d", { willReadFrequently: true });
const edgesCtx = edgesCanvas.getContext("2d", { willReadFrequently: true });
const overlayCtx = overlayCanvas.getContext("2d", { willReadFrequently: true });

let loadedImage = null;

imageInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => {
    const image = new Image();
    image.onload = () => {
      loadedImage = image;
      renderSourceImage(image);
      analyzeButton.disabled = false;
      statusPill.textContent = "Imagem carregada";
      resetMetrics();
    };
    image.src = reader.result;
  };
  reader.readAsDataURL(file);
});

analyzeButton.addEventListener("click", () => {
  if (!loadedImage) return;
  statusPill.textContent = "Processando";
  window.requestAnimationFrame(runAnalysis);
});

function renderSourceImage(image) {
  const target = fitDimensions(image.width, image.height, 960, 720);
  [originalCanvas, processedCanvas, edgesCanvas, overlayCanvas].forEach((canvas) => {
    canvas.width = target.width;
    canvas.height = target.height;
  });

  originalCtx.clearRect(0, 0, target.width, target.height);
  originalCtx.drawImage(image, 0, 0, target.width, target.height);

  processedCtx.clearRect(0, 0, target.width, target.height);
  edgesCtx.clearRect(0, 0, target.width, target.height);
  overlayCtx.clearRect(0, 0, target.width, target.height);
}

function runAnalysis() {
  const { width, height } = originalCanvas;
  const source = originalCtx.getImageData(0, 0, width, height);
  const grayscale = toGrayscale(source.data);
  const blurred = boxBlur(grayscale, width, height);
  const edgeData = detectEdges(blurred, width, height);
  const detections = findConnectedRegions(edgeData.binary, width, height);

  drawGrayscale(processedCtx, blurred, width, height);
  drawBinary(edgesCtx, edgeData.binary, width, height);
  drawOverlay(source, detections, width, height);
  updateMetrics(source.data, blurred, detections, width, height);
}

function toGrayscale(data) {
  const gray = new Uint8ClampedArray(data.length / 4);
  for (let i = 0, j = 0; i < data.length; i += 4, j += 1) {
    gray[j] = Math.round(data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114);
  }
  return gray;
}

function boxBlur(gray, width, height) {
  const result = new Uint8ClampedArray(gray.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let sum = 0;
      let count = 0;
      for (let ky = -1; ky <= 1; ky += 1) {
        for (let kx = -1; kx <= 1; kx += 1) {
          const px = x + kx;
          const py = y + ky;
          if (px >= 0 && py >= 0 && px < width && py < height) {
            sum += gray[py * width + px];
            count += 1;
          }
        }
      }
      result[y * width + x] = sum / count;
    }
  }
  return result;
}

function detectEdges(gray, width, height) {
  const magnitude = new Uint8ClampedArray(gray.length);
  const binary = new Uint8ClampedArray(gray.length);
  let maxMag = 0;

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const i = y * width + x;
      const gx =
        -gray[(y - 1) * width + (x - 1)] + gray[(y - 1) * width + (x + 1)] +
        -2 * gray[i - 1] + 2 * gray[i + 1] +
        -gray[(y + 1) * width + (x - 1)] + gray[(y + 1) * width + (x + 1)];
      const gy =
        -gray[(y - 1) * width + (x - 1)] - 2 * gray[(y - 1) * width + x] - gray[(y - 1) * width + (x + 1)] +
        gray[(y + 1) * width + (x - 1)] + 2 * gray[(y + 1) * width + x] + gray[(y + 1) * width + (x + 1)];

      const mag = Math.min(255, Math.hypot(gx, gy));
      magnitude[i] = mag;
      if (mag > maxMag) maxMag = mag;
    }
  }

  const threshold = Math.max(35, maxMag * 0.24);
  for (let i = 0; i < magnitude.length; i += 1) {
    binary[i] = magnitude[i] >= threshold ? 255 : 0;
  }

  return { magnitude, binary };
}

function findConnectedRegions(binary, width, height) {
  const visited = new Uint8Array(binary.length);
  const regions = [];
  const minimumArea = Math.max(180, Math.floor(width * height * 0.0025));

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const start = y * width + x;
      if (visited[start] || binary[start] === 0) continue;

      const queue = [start];
      visited[start] = 1;

      let head = 0;
      let area = 0;
      let minX = x;
      let minY = y;
      let maxX = x;
      let maxY = y;

      while (head < queue.length) {
        const current = queue[head++];
        const cx = current % width;
        const cy = Math.floor(current / width);

        area += 1;
        if (cx < minX) minX = cx;
        if (cy < minY) minY = cy;
        if (cx > maxX) maxX = cx;
        if (cy > maxY) maxY = cy;

        for (let ny = cy - 1; ny <= cy + 1; ny += 1) {
          for (let nx = cx - 1; nx <= cx + 1; nx += 1) {
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
            const next = ny * width + nx;
            if (visited[next] || binary[next] === 0) continue;
            visited[next] = 1;
            queue.push(next);
          }
        }
      }

      if (area >= minimumArea) {
        const boxWidth = maxX - minX + 1;
        const boxHeight = maxY - minY + 1;
        const aspect = boxWidth / Math.max(boxHeight, 1);
        regions.push({
          area,
          minX,
          minY,
          maxX,
          maxY,
          boxWidth,
          boxHeight,
          aspect,
          label: classifyRegion(area, aspect, boxWidth, boxHeight, width, height)
        });
      }
    }
  }

  return regions.sort((a, b) => b.area - a.area).slice(0, 8);
}

function classifyRegion(area, aspect, boxWidth, boxHeight, width, height) {
  const relativeArea = area / (width * height);
  if (relativeArea > 0.12) return "Regiao dominante";
  if (aspect > 2.2) return "Elemento horizontal";
  if (aspect < 0.55) return "Elemento vertical";
  if (Math.abs(boxWidth - boxHeight) < Math.min(boxWidth, boxHeight) * 0.22) return "Forma quase quadrada";
  return "Objeto irregular";
}

function drawGrayscale(ctx, gray, width, height) {
  const imageData = ctx.createImageData(width, height);
  for (let i = 0, j = 0; j < gray.length; i += 4, j += 1) {
    imageData.data[i] = gray[j];
    imageData.data[i + 1] = gray[j];
    imageData.data[i + 2] = gray[j];
    imageData.data[i + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

function drawBinary(ctx, binary, width, height) {
  const imageData = ctx.createImageData(width, height);
  for (let i = 0, j = 0; j < binary.length; i += 4, j += 1) {
    imageData.data[i] = binary[j];
    imageData.data[i + 1] = binary[j];
    imageData.data[i + 2] = binary[j];
    imageData.data[i + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

function drawOverlay(source, detections, width, height) {
  overlayCtx.putImageData(source, 0, 0);
  overlayCtx.lineWidth = 3;
  overlayCtx.font = '700 14px "Manrope", sans-serif';

  detections.forEach((region, index) => {
    overlayCtx.strokeStyle = "#f97316";
    overlayCtx.fillStyle = "rgba(15, 118, 110, 0.92)";
    overlayCtx.strokeRect(region.minX, region.minY, region.boxWidth, region.boxHeight);

    const label = `${index + 1}. ${region.label}`;
    const textWidth = overlayCtx.measureText(label).width;
    const tagWidth = textWidth + 18;
    const tagHeight = 26;
    const tagY = Math.max(0, region.minY - tagHeight - 6);

    overlayCtx.fillRect(region.minX, tagY, tagWidth, tagHeight);
    overlayCtx.fillStyle = "#ffffff";
    overlayCtx.fillText(label, region.minX + 9, tagY + 17);
  });
}

function updateMetrics(sourceData, gray, detections, width, height) {
  const totalPixels = width * height;
  let graySum = 0;
  let rSum = 0;
  let gSum = 0;
  let bSum = 0;
  let occupiedArea = 0;

  for (let i = 0, px = 0; i < sourceData.length; i += 4, px += 1) {
    graySum += gray[px];
    rSum += sourceData[i];
    gSum += sourceData[i + 1];
    bSum += sourceData[i + 2];
  }

  detections.forEach((region) => {
    occupiedArea += region.area;
  });

  const avgGray = Math.round(graySum / totalPixels);
  const avgColor = {
    r: Math.round(rSum / totalPixels),
    g: Math.round(gSum / totalPixels),
    b: Math.round(bSum / totalPixels)
  };

  objectsCount.textContent = String(detections.length);
  coverageValue.textContent = `${((occupiedArea / totalPixels) * 100).toFixed(1)}%`;
  intensityValue.textContent = String(avgGray);
  colorValue.textContent = describeColor(avgColor);
  statusPill.textContent = detections.length ? "Analise concluida" : "Sem elementos relevantes";

  detectedList.innerHTML = "";
  if (!detections.length) {
    detectedList.innerHTML = "<li>Nenhum elemento com area minima foi encontrado.</li>";
  } else {
    detections.forEach((region, index) => {
      const item = document.createElement("li");
      item.textContent = `${index + 1}. ${region.label} | area ${region.area}px | caixa ${region.boxWidth}x${region.boxHeight}`;
      detectedList.appendChild(item);
    });
  }

  analysisSummary.textContent = buildSummary(detections, avgGray, avgColor, occupiedArea, totalPixels);
}

function describeColor({ r, g, b }) {
  if (r > g + 20 && r > b + 20) return "Vermelho";
  if (g > r + 20 && g > b + 20) return "Verde";
  if (b > r + 20 && b > g + 20) return "Azul";
  if (r > 190 && g > 190 && b > 190) return "Clara";
  if (r < 80 && g < 80 && b < 80) return "Escura";
  return "Neutra";
}

function buildSummary(detections, avgGray, avgColor, occupiedArea, totalPixels) {
  const dominantColor = describeColor(avgColor).toLowerCase();
  const density = ((occupiedArea / totalPixels) * 100).toFixed(1);

  if (!detections.length) {
    return `A imagem foi normalizada e analisada, mas nao apresentou regioes suficientemente fortes para a segmentacao atual. A intensidade media ficou em ${avgGray} e a leitura geral de cor foi ${dominantColor}.`;
  }

  const primary = detections[0].label.toLowerCase();
  return `Foram encontrados ${detections.length} elementos principais. O destaque visual da cena e ${primary}, com ocupacao aproximada de ${density}% da area analisada. A iluminacao media ficou em ${avgGray} e a cor predominante da imagem foi classificada como ${dominantColor}.`;
}

function fitDimensions(width, height, maxWidth, maxHeight) {
  const scale = Math.min(maxWidth / width, maxHeight / height, 1);
  return {
    width: Math.round(width * scale),
    height: Math.round(height * scale)
  };
}

function resetMetrics() {
  objectsCount.textContent = "0";
  coverageValue.textContent = "0%";
  intensityValue.textContent = "0";
  colorValue.textContent = "-";
  detectedList.innerHTML = "<li>Imagem pronta para analise.</li>";
  analysisSummary.textContent = "A imagem foi carregada. Clique em analisar para aplicar o pre-processamento e procurar regioes de interesse.";
}
