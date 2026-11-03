# ContaMoedas BR (iOS)

<p align="center">
  <img src="https://img.shields.io/badge/Platform-iOS%2017.0%2B-blue?style=for-the-badge&logo=apple" alt="iOS 17.0+" />
  <img src="https://img.shields.io/badge/Swift-5.0-orange?style=for-the-badge&logo=swift" alt="Swift 5.0" />
  <img src="https://img.shields.io/badge/Framework-SwiftUI%20%7C%20Vision%20%7C%20CoreML-green?style=for-the-badge" alt="Frameworks" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge" alt="License" />
</p>

---

## 🇧🇷 Português (Brasil)

### Visão Geral
O **ContaMoedas BR** é um aplicativo nativo para iOS desenvolvido em **SwiftUI**, **Vision** e **CoreML** capaz de detectar, classificar e somar moedas do Real Brasileiro (**R$ 0,05, R$ 0,10, R$ 0,25, R$ 0,50 e R$ 1,00**) em alta velocidade utilizando a câmera do iPhone.

O sistema opera **100% offline (On-Device AI)** na Apple Neural Engine, combinando processamento geométrico radial em 360°, filtros cromáticos de superfície metálica e um modelo local de aprendizado de máquina (`CoinClassifier.mlmodel`) treinado para reconhecer as moedas brasileiras tanto com o valor numérico visível (reverso) quanto com as efígies históricas (anverso).

---

### Funcionalidades Principais

1. **Modo 1: Tirar Foto e Identificar (IA CoreML)**
   - Captura a imagem em alta resolução.
   - Exibe indicador de processamento em tempo real.
   - Congela o resultado na tela marcando o valor individual de cada moeda detectada e somando ao total acumulado.
   - Botão para retomar o vídeo da câmera ao vivo a qualquer momento.

2. **Modo 2: Contagem em Tempo Real (Sweep Contínuo)**
   - Permite passar a câmera continuamente sobre superfícies com moedas espalhadas.
   - Sistema de rastreamento e persistência espacial com memória temporal que impede a contagem duplicada da mesma moeda.

3. **Modo 3: Modo Manual (Mira Central 1x)**
   - Escaneia individualmente a moeda posicionada na mira tracejada central, adicionando uma unidade por toque.

4. **Classificação Invariante à Face da Moeda (Cara ou Coroa)**
   - Utiliza as especificações oficiais de diâmetro e metalurgia do Banco Central do Brasil para separar com precisão física absoluta moedas de tamanhos diferentes:
     - **R$ 1,00:** 27,0 mm — Bimetálica (anel de bronze dourado e núcleo de aço inox prateado).
     - **R$ 0,50:** 23,0 mm — Aço inoxidável prateado espesso.
     - **R$ 0,25:** 25,0 mm — Aço revestido de bronze dourado (raio 25% maior que a de 10¢).
     - **R$ 0,10:** 20,0 mm — Aço revestido de bronze dourado (menor diâmetro).
     - **R$ 0,05:** 22,0 mm — Aço revestido de cobre avermelhado.
     - **R$ 0,10 (1994):** 22,0 mm — Aço inoxidável prateado da 1ª família.

5. **Proteções Avançadas contra Falsos Positivos**
   - **Filtro Anti-Pele Humana:** Descarta dedos e palmas das mãos comparando razões cromáticas `(R - G) / (G - B)`.
   - **Verificação Circular em 360°:** Avalia a convergência de vetores de gradiente radial em múltiplos setores da circunferência.
   - **Rejeição de Fundo:** Ignora listras, toalhas, mesas e texturas sem assinatura metálica real.

---

### Arquitetura de Software

```
ContaMoedas/
├── Sources/
│   ├── ContaMoedasApp.swift        # Entrada do App, Motor de Visão e Interface SwiftUI
│   └── CoinClassifier.mlmodel      # Modelo CoreML treinado com anverso/reverso e classes de fundo
├── Training/
│   └── Dataset/                    # Amostras rotuladas por denominação (5c, 10c, 25c, 50c, 1R, fundo)
├── Info.plist                      # Permissões de câmera (AVCapture)
└── project.yml                     # Especificação declarativa do projeto para XcodeGen
```

---

### Como Compilar e Rodar

#### Pré-requisitos
- macOS com **Xcode 15+** instalado.
- Ferramenta [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`).
- iPhone com iOS 17.0 ou superior (ou simulador com suporte a câmera).

#### Passos
```bash
# 1. Clone o repositório
git clone https://github.com/mauriciomholiveira/ContaMoedas.git
cd ContaMoedas

# 2. Gere o projeto Xcode via XcodeGen
xcodegen generate

# 3. Abra o projeto no Xcode
open ContaMoedas.xcodeproj
```

---

## 🇺🇸 English

### Overview
**ContaMoedas BR** is a native iOS application built with **SwiftUI**, **Vision**, and **CoreML** designed to detect, classify, and count Brazilian Real coins (**5¢, 10¢, 25¢, 50¢, and R$ 1**) in real time using the iPhone camera.

Operating **100% offline (On-Device AI)** powered by the Apple Neural Engine, it combines 360° radial symmetry edge verification, chromatic metallic surface segmentation, and a CoreML classifier (`CoinClassifier.mlmodel`) trained on both coin values (reverse) and historical effigies (obverse).

---

### Key Features

1. **Photo Mode (Local CoreML Inference)**
   - High-definition single-frame capture with animated on-device inference state.
   - Visual overlay displaying detected bounding disks and values.
   - Automatic accumulator with an instant live-camera resume button.

2. **Real-Time Continuous Counting Mode**
   - Live camera sweep mode with spatial tracking to prevent duplicate counting as coins move across the viewport.

3. **Manual Reticle Scan Mode (1x per tap)**
   - Center-target reticle allows precision one-by-one verification.

4. **Face-Invariant Physical Dimension Calibration**
   - Based on official Central Bank of Brazil mint specifications:
     - **R$ 1.00:** 27.0 mm (Bimetallic: golden outer bronze ring + stainless steel core).
     - **R$ 0.50:** 23.0 mm (Stainless steel monometallic).
     - **R$ 0.25:** 25.0 mm (Bronze-coated steel, 25% larger radius than 10¢).
     - **R$ 0.10:** 20.0 mm (Bronze-coated steel).
     - **R$ 0.05:** 22.0 mm (Copper-coated steel with reddish tone).
     - **R$ 0.10 (1994):** 22.0 mm (1st family stainless steel).

5. **Strict Anti-False-Positive Pipeline**
   - **Skin Tone Filter:** Rejects hands and fingers based on chromatic channel ratios.
   - **Radial Hough Symmetry:** Validates true circular rims across 24 radial sectors.
   - **Background Discrimination:** Automatically discards non-metallic cloths, tablecloth patterns, and room artifacts.

---

### License
Released under the [MIT License](LICENSE).
