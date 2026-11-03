import SwiftUI
import AVFoundation
import Vision
import CoreML
import UIKit

@main
struct ContaMoedasApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(.dark)
        }
    }
}

// MARK: - Modelos de Moedas do Brasil (Real - R$)
enum CoinDenomination: Int, CaseIterable, Identifiable, Codable {
    case fiveCents = 5
    case tenCents = 10
    case twentyFiveCents = 25
    case fiftyCents = 50
    case oneReal = 100

    var id: Int { rawValue }

    var label: String {
        switch self {
        case .fiveCents: return "5¢"
        case .tenCents: return "10¢"
        case .twentyFiveCents: return "25¢"
        case .fiftyCents: return "50¢"
        case .oneReal: return "R$ 1"
        }
    }

    var fullName: String {
        switch self {
        case .fiveCents: return "5 Centavos"
        case .tenCents: return "10 Centavos"
        case .twentyFiveCents: return "25 Centavos"
        case .fiftyCents: return "50 Centavos"
        case .oneReal: return "1 Real"
        }
    }

    var valueInReais: Double {
        Double(rawValue) / 100.0
    }

    var badgeColor: Color {
        switch self {
        case .fiveCents:
            return Color(red: 0.78, green: 0.44, blue: 0.25) // Cobre
        case .tenCents:
            return Color(red: 0.90, green: 0.72, blue: 0.20) // Dourado
        case .twentyFiveCents:
            return Color(red: 0.85, green: 0.65, blue: 0.22) // Bronze Dourado
        case .fiftyCents:
            return Color(red: 0.75, green: 0.78, blue: 0.82) // Prateado Inox
        case .oneReal:
            return Color(red: 0.95, green: 0.80, blue: 0.25) // Bimetálica
        }
    }

    var ringColor: Color {
        switch self {
        case .oneReal: return Color(red: 0.85, green: 0.68, blue: 0.18)
        case .fiftyCents: return Color.white.opacity(0.85)
        default: return badgeColor.opacity(0.9)
        }
    }

    static func fromMLIdentifier(_ id: String) -> CoinDenomination? {
        switch id {
        case "5_centavos": return .fiveCents
        case "10_centavos": return .tenCents
        case "25_centavos": return .twentyFiveCents
        case "50_centavos": return .fiftyCents
        case "1_real": return .oneReal
        default: return nil
        }
    }
}

struct DetectedCoin: Identifiable {
    let id = UUID()
    let denomination: CoinDenomination
    let boundingBox: CGRect // Normalizado (0..1, origem no topo-esquerdo da visualização)
    let confidence: Float
    let center: CGPoint
}

struct LearnedCoinSample: Codable {
    let denominationRaw: Int
    let coreR: Float
    let coreG: Float
    let coreB: Float
    let rimR: Float
    let rimG: Float
    let rimB: Float
    let radiusRatio: Float
}

// MARK: - Motor de Visão Computacional + IA Local CoreML (CoinClassifier)
final class CoinVisionEngine: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate, AVCapturePhotoCaptureDelegate {
    @Published var detectedCoins: [DetectedCoin] = []
    @Published var isAutoCountingSweep: Bool = false
    @Published var statusMessage: String = "IA Local CoreML pronta: Foto, Tempo Real ou Manual"
    @Published var counts: [CoinDenomination: Int] = [
        .fiveCents: 0,
        .tenCents: 0,
        .twentyFiveCents: 0,
        .fiftyCents: 0,
        .oneReal: 0
    ]
    @Published var torchOn: Bool = false
    @Published var learnedSamplesCount: Int = 0
    @Published var capturedSnapshot: UIImage? = nil
    @Published var isProcessingPhoto: Bool = false

    let session = AVCaptureSession()
    private let photoOutput = AVCapturePhotoOutput()
    private let visionQueue = DispatchQueue(label: "com.mauricio.contamoedas.vision", qos: .userInitiated)
    private var lastProcessedTime: CFTimeInterval = 0
    private var latestPixelBuffer: CVPixelBuffer?
    private let bufferLock = NSLock()

    // Modelo de IA Local CoreML treinado para moedas brasileiras (5c, 10c, 25c, 50c, 1 real e fundo)
    private var vnCoinModel: VNCoreMLModel?

    // Rastreamento espacial para o modo "Tempo Real"
    private var countedSpatialPoints: [(point: CGPoint, time: CFTimeInterval, denom: CoinDenomination)] = []
    private var candidateStability: [(point: CGPoint, hits: Int, denom: CoinDenomination, lastSeen: CFTimeInterval)] = []

    // Amostras extras aprendidas pelo usuário no aparelho
    private var learnedSamples: [LearnedCoinSample] = []
    private let storageKey = "LearnedCoinVectorsV3"

    override init() {
        super.init()
        loadCoreMLModel()
        loadLearnedSamples()
        configureCamera()
    }

    private func loadCoreMLModel() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            let wrapper = try CoinClassifier(configuration: config)
            vnCoinModel = try VNCoreMLModel(for: wrapper.model)
        } catch {
            if let url = Bundle.main.url(forResource: "CoinClassifier", withExtension: "mlmodelc"),
               let mlModel = try? MLModel(contentsOf: url),
               let vnModel = try? VNCoreMLModel(for: mlModel) {
                vnCoinModel = vnModel
            }
        }
    }

    var totalReais: Double {
        counts.reduce(0.0) { partial, pair in
            partial + (Double(pair.value) * pair.key.valueInReais)
        }
    }

    var visibleTotalReais: Double {
        detectedCoins.reduce(0.0) { $0 + $1.denomination.valueInReais }
    }

    var totalCoinsCount: Int {
        counts.values.reduce(0, +)
    }

    func configureCamera() {
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else {
            DispatchQueue.main.async {
                self.statusMessage = "Câmera indisponível."
            }
            session.commitConfiguration()
            return
        }

        if session.canAddInput(input) {
            session.addInput(input)
        }

        try? device.lockForConfiguration()
        if device.isFocusModeSupported(.continuousAutoFocus) {
            device.focusMode = .continuousAutoFocus
        }
        if device.isExposureModeSupported(.continuousAutoExposure) {
            device.exposureMode = .continuousAutoExposure
        }
        if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
            device.whiteBalanceMode = .continuousAutoWhiteBalance
        }
        device.unlockForConfiguration()

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: visionQueue)

        if session.canAddOutput(output) {
            session.addOutput(output)
        }

        if session.canAddOutput(photoOutput) {
            session.addOutput(photoOutput)
        }

        if let connection = output.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
        }

        session.commitConfiguration()

        visionQueue.async { [weak self] in
            self?.session.startRunning()
        }
    }

    func toggleTorch() {
        guard let device = AVCaptureDevice.default(for: .video), device.hasTorch else { return }
        do {
            try device.lockForConfiguration()
            torchOn.toggle()
            device.torchMode = torchOn ? .on : .off
            device.unlockForConfiguration()
        } catch {}
    }

    // MARK: - Botão 1: TIRAR FOTO E IDENTIFICAR (IA LOCAL)
    func capturePhotoAndIdentify() {
        isAutoCountingSweep = false

        bufferLock.lock()
        let currentBuffer = latestPixelBuffer
        bufferLock.unlock()

        if let pixelBuffer = currentBuffer,
           let cgImage = createCGImage(from: pixelBuffer) {
            // 1. Primeiro congela a foto capturada na tela imediatamente e mostra o loading da IA
            let uiImg = UIImage(cgImage: cgImage)
            capturedSnapshot = uiImg
            detectedCoins = []
            isProcessingPhoto = true
            statusMessage = "Analisando foto com IA Local CoreML..."
            UIImpactFeedbackGenerator(style: .medium).impactOccurred()

            // 2. Executa a análise profunda em background mantendo o overlay de loading visível
            visionQueue.async { [weak self] in
                guard let self = self else { return }
                let start = CACurrentMediaTime()
                let coins = self.detectAndClassifyCoinsWithLocalAI(in: cgImage)
                let elapsed = CACurrentMediaTime() - start
                let remainingDelay = max(0.0, 0.45 - elapsed)

                DispatchQueue.main.asyncAfter(deadline: .now() + remainingDelay) {
                    self.isProcessingPhoto = false
                    self.detectedCoins = coins
                    if coins.isEmpty {
                        self.statusMessage = "Nenhuma moeda detectada na foto."
                        UINotificationFeedbackGenerator().notificationOccurred(.warning)
                    } else {
                        for coin in coins {
                            self.counts[coin.denomination, default: 0] += 1
                        }
                        let addedValue = coins.reduce(0.0) { $0 + $1.denomination.valueInReais }
                        self.statusMessage = String(format: "Foto IA: +%d moedas (+R$ %.2f)", coins.count, addedValue)
                            .replacingOccurrences(of: ".", with: ",")
                        UINotificationFeedbackGenerator().notificationOccurred(.success)
                    }
                }
            }
            return
        }

        isProcessingPhoto = true
        statusMessage = "Capturando foto..."
        let settings = AVCapturePhotoSettings()
        photoOutput.capturePhoto(with: settings, delegate: self)
    }

    func dismissSnapshot() {
        capturedSnapshot = nil
        statusMessage = "Câmera ao vivo retomada."
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        guard error == nil,
              let data = photo.fileDataRepresentation(),
              let uiImage = UIImage(data: data),
              let cgImage = uiImage.cgImage else {
            DispatchQueue.main.async {
                self.isProcessingPhoto = false
                self.statusMessage = "Erro ao capturar foto."
            }
            return
        }

        visionQueue.async { [weak self] in
            guard let self = self else { return }
            let coins = self.detectAndClassifyCoinsWithLocalAI(in: cgImage)
            DispatchQueue.main.async {
                self.isProcessingPhoto = false
                self.capturedSnapshot = uiImage
                self.detectedCoins = coins
                for coin in coins {
                    self.counts[coin.denomination, default: 0] += 1
                }
                let addedValue = coins.reduce(0.0) { $0 + $1.denomination.valueInReais }
                self.statusMessage = String(format: "Foto IA: +%d moedas (+R$ %.2f)", coins.count, addedValue)
                    .replacingOccurrences(of: ".", with: ",")
                UINotificationFeedbackGenerator().notificationOccurred(.success)
            }
        }
    }

    // MARK: - Botão 2: EM TEMPO REAL (Contagem Automática Contínua)
    func toggleRealtimeMode() {
        if capturedSnapshot != nil {
            capturedSnapshot = nil
        }
        isAutoCountingSweep.toggle()
        if isAutoCountingSweep {
            countedSpatialPoints.removeAll()
            candidateStability.removeAll()
            statusMessage = "TEMPO REAL ATIVO: Passe a câmera sobre as moedas..."
            UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
        } else {
            statusMessage = "Modo Tempo Real pausado."
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
        }
    }

    // MARK: - Botão 3: MANUAL (1 Clique por Vez na Moeda da Mira)
    func scanSingleCoinManual() {
        if capturedSnapshot != nil {
            capturedSnapshot = nil
        }
        let aimCenter = CGPoint(x: 0.5, y: 0.5)
        if let bestCoin = detectedCoins.min(by: {
            hypot($0.center.x - aimCenter.x, $0.center.y - aimCenter.y) <
            hypot($1.center.x - aimCenter.x, $1.center.y - aimCenter.y)
        }) {
            counts[bestCoin.denomination, default: 0] += 1
            statusMessage = "+1 moeda de \(bestCoin.denomination.fullName) adicionada (Manual)!"
            UIImpactFeedbackGenerator(style: .rigid).impactOccurred()
        } else {
            statusMessage = "Posicione uma moeda na mira central ou aproxime a câmera."
            UINotificationFeedbackGenerator().notificationOccurred(.warning)
        }
    }

    func incrementManual(_ coin: CoinDenomination) {
        counts[coin, default: 0] += 1
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }

    func decrementManual(_ coin: CoinDenomination) {
        if let current = counts[coin], current > 0 {
            counts[coin] = current - 1
            UIImpactFeedbackGenerator(style: .soft).impactOccurred()
        }
    }

    func resetAll() {
        for key in counts.keys {
            counts[key] = 0
        }
        countedSpatialPoints.removeAll()
        candidateStability.removeAll()
        capturedSnapshot = nil
        statusMessage = "Contador zerado."
        UINotificationFeedbackGenerator().notificationOccurred(.warning)
    }

    // MARK: - Calibração Extra On-Device
    func learnCoinFromCamera(target: CoinDenomination) {
        bufferLock.lock()
        guard let pb = latestPixelBuffer else {
            bufferLock.unlock()
            statusMessage = "Aguardando imagem da câmera..."
            return
        }
        let sampleOpt = extractCenterCoinSample(from: pb, target: target)
        bufferLock.unlock()

        guard let sample = sampleOpt else {
            statusMessage = "Nenhuma moeda metálica detectada no centro para calibrar."
            UINotificationFeedbackGenerator().notificationOccurred(.error)
            return
        }

        learnedSamples.append(sample)
        saveLearnedSamples()
        learnedSamplesCount = learnedSamples.count
        statusMessage = "IA calibrada com amostra de \(target.fullName)!"
        UINotificationFeedbackGenerator().notificationOccurred(.success)
    }

    func clearLearnedSamples() {
        learnedSamples.removeAll()
        UserDefaults.standard.removeObject(forKey: storageKey)
        learnedSamplesCount = 0
        statusMessage = "Calibração extra limpa (IA CoreML original ativa)."
        UINotificationFeedbackGenerator().notificationOccurred(.warning)
    }

    private func saveLearnedSamples() {
        if let data = try? JSONEncoder().encode(learnedSamples) {
            UserDefaults.standard.set(data, forKey: storageKey)
        }
    }

    private func loadLearnedSamples() {
        if let data = UserDefaults.standard.data(forKey: storageKey),
           let decoded = try? JSONDecoder().decode([LearnedCoinSample].self, from: data) {
            learnedSamples = decoded
            learnedSamplesCount = decoded.count
        }
    }

    // MARK: - Loop de Captura de Vídeo em Tempo Real
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        bufferLock.lock()
        latestPixelBuffer = pixelBuffer
        bufferLock.unlock()

        // Se o usuário tirou uma foto estática e está visualizando o resultado, pausa a sobreposição ao vivo
        if capturedSnapshot != nil || isProcessingPhoto {
            return
        }

        let now = CACurrentMediaTime()
        guard now - lastProcessedTime >= 0.20 else { return }
        lastProcessedTime = now

        guard let cgImage = createCGImage(from: pixelBuffer) else { return }
        let results = detectAndClassifyCoinsWithLocalAI(in: cgImage)

        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            if self.capturedSnapshot != nil { return }
            self.detectedCoins = results
            if self.isAutoCountingSweep {
                self.processRealtimeTracking(with: results, timestamp: now)
            }
        }
    }

    private func processRealtimeTracking(with currentCoins: [DetectedCoin], timestamp: CFTimeInterval) {
        countedSpatialPoints.removeAll { timestamp - $0.time > 4.0 }
        candidateStability.removeAll { timestamp - $0.lastSeen > 0.9 }

        for coin in currentCoins {
            var matchedStabilityIndex: Int? = nil
            for i in candidateStability.indices {
                let dist = hypot(candidateStability[i].point.x - coin.center.x,
                                 candidateStability[i].point.y - coin.center.y)
                if dist < 0.08 {
                    matchedStabilityIndex = i
                    break
                }
            }

            if let idx = matchedStabilityIndex {
                candidateStability[idx].point = coin.center
                candidateStability[idx].hits += 1
                candidateStability[idx].denom = coin.denomination
                candidateStability[idx].lastSeen = timestamp

                if candidateStability[idx].hits >= 2 {
                    let alreadyCounted = countedSpatialPoints.contains { tracked in
                        hypot(tracked.point.x - coin.center.x, tracked.point.y - coin.center.y) < 0.11
                    }
                    if !alreadyCounted {
                        counts[coin.denomination, default: 0] += 1
                        countedSpatialPoints.append((point: coin.center, time: timestamp, denom: coin.denomination))
                        statusMessage = "+1 \(coin.denomination.label) somado em Tempo Real!"
                        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                    }
                }
            } else {
                candidateStability.append((point: coin.center, hits: 1, denom: coin.denomination, lastSeen: timestamp))
            }
        }
    }

    // MARK: - Detector Circular de Simetria Radial (Hough) + Classificador IA Local CoreML
    private struct CircleCandidate {
        let x: Int
        let y: Int
        let r: Int
        let score: Float
        let coreR: Float
        let coreG: Float
        let coreB: Float
        let rimR: Float
        let rimG: Float
        let rimB: Float
        let warmth: Float
        let maxDiff: Float
    }

    private func detectAndClassifyCoinsWithLocalAI(in cgImage: CGImage) -> [DetectedCoin] {
        let W = 384
        let H = max(256, Int(Double(cgImage.height) * Double(W) / Double(cgImage.width)))
        var pixels = [UInt8](repeating: 0, count: W * H * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: &pixels,
            width: W,
            height: H,
            bitsPerComponent: 8,
            bytesPerRow: W * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return []
        }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: W, height: H))

        // 1. Luminância + suavização 3x3
        var luma = [Float](repeating: 0, count: W * H)
        for i in 0..<(W * H) {
            let r = Float(pixels[i * 4])
            let g = Float(pixels[i * 4 + 1])
            let b = Float(pixels[i * 4 + 2])
            luma[i] = 0.299 * r + 0.587 * g + 0.114 * b
        }

        var blur = luma
        for y in 1..<(H - 1) {
            for x in 1..<(W - 1) {
                var sum: Float = 0
                for dy in -1...1 {
                    for dx in -1...1 {
                        sum += luma[(y + dy) * W + (x + dx)]
                    }
                }
                blur[y * W + x] = sum / 9.0
            }
        }

        // 2. Gradientes de Sobel
        var gx = [Float](repeating: 0, count: W * H)
        var gy = [Float](repeating: 0, count: W * H)
        var mag = [Float](repeating: 0, count: W * H)
        for y in 1..<(H - 1) {
            for x in 1..<(W - 1) {
                let sx = -blur[(y - 1) * W + (x - 1)] + blur[(y - 1) * W + (x + 1)]
                       - 2 * blur[y * W + (x - 1)]   + 2 * blur[y * W + (x + 1)]
                       - blur[(y + 1) * W + (x - 1)] + blur[(y + 1) * W + (x + 1)]
                let sy = -blur[(y - 1) * W + (x - 1)] - 2 * blur[(y - 1) * W + x] - blur[(y - 1) * W + (x + 1)]
                       + blur[(y + 1) * W + (x - 1)] + 2 * blur[(y + 1) * W + x] + blur[(y + 1) * W + (x + 1)]
                let m = sqrt(sx * sx + sy * sy)
                gx[y * W + x] = sx
                gy[y * W + x] = sy
                mag[y * W + x] = m
            }
        }

        // 3. Acumulador Radial Hough para raios de moedas em diferentes distâncias da câmera (13..42 px em W=384)
        let minR = 13
        let maxR = 40
        var acc = [Float](repeating: 0, count: W * H)
        var bestR = [Int](repeating: 18, count: W * H)

        for r in minR...maxR {
            var rAcc = [Float](repeating: 0, count: W * H)
            for y in r..<(H - r) {
                for x in r..<(W - r) {
                    let idx = y * W + x
                    let m = mag[idx]
                    if m < 44 { continue }
                    let ux = gx[idx] / m
                    let uy = gy[idx] / m
                    for sign in [-1.0 as Float, 1.0 as Float] {
                        let cx = Int(round(Float(x) + sign * ux * Float(r)))
                        let cy = Int(round(Float(y) + sign * uy * Float(r)))
                        if cx >= r && cx < W - r && cy >= r && cy < H - r {
                            rAcc[cy * W + cx] += m
                        }
                    }
                }
            }
            for y in r..<(H - r) {
                for x in r..<(W - r) {
                    var s: Float = 0
                    for dy in -1...1 {
                        for dx in -1...1 {
                            s += rAcc[(y + dy) * W + (x + dx)]
                        }
                    }
                    let normScore = s / Float(r)
                    if normScore > acc[y * W + x] {
                        acc[y * W + x] = normScore
                        bestR[y * W + x] = r
                    }
                }
            }
        }

        // 4. Supressão de não-máximos + verificação geométrica 360° (24 pontos) + simetria de quadrantes
        var candidates: [CircleCandidate] = []
        for y in 18..<(H - 18) {
            for x in 18..<(W - 18) {
                let s = acc[y * W + x]
                // Limiar ajustado para 120 (detecta moedas com rosto/efígie e relevo sem perder sensibilidade)
                if s < 120 { continue }
                var isMax = true
                for dy in -9...9 {
                    for dx in -9...9 {
                        if acc[(y + dy) * W + (x + dx)] > s {
                            isMax = false
                            break
                        }
                    }
                    if !isMax { break }
                }
                if !isMax { continue }
                let r = bestR[y * W + x]

                // 4.1 Verificação de Borda Circular Real em 24 ângulos ao redor do perímetro
                var alignedEdges = 0
                for k in 0..<24 {
                    let ang = Float(k) * (2.0 * .pi / 24.0)
                    let csA = cos(ang)
                    let snA = sin(ang)
                    var bestDot: Float = 0
                    var bestM: Float = 0
                    for dr in -1...1 {
                        let px = Int(round(Float(x) + Float(r + dr) * csA))
                        let py = Int(round(Float(y) + Float(r + dr) * snA))
                        if px >= 1 && px < W - 1 && py >= 1 && py < H - 1 {
                            let m = mag[py * W + px]
                            if m > bestM {
                                bestM = m
                                bestDot = abs(gx[py * W + px] * csA + gy[py * W + px] * snA) / max(m, 1.0)
                            }
                        }
                    }
                    if bestM > 35.0 && bestDot > 0.65 {
                        alignedEdges += 1
                    }
                }
                // Exige pelo menos 8 de 24 setores (permite moedas encostadas em outras na mesa)
                if alignedEdges < 8 { continue }

                var coreR: Float = 0, coreG: Float = 0, coreB: Float = 0, coreCnt: Float = 0
                var rimR: Float = 0, rimG: Float = 0, rimB: Float = 0, rimCnt: Float = 0
                var outR: Float = 0, outG: Float = 0, outB: Float = 0, outCnt: Float = 0
                var qR = [Float](repeating: 0, count: 4)
                var qG = [Float](repeating: 0, count: 4)
                var qB = [Float](repeating: 0, count: 4)
                var qN = [Float](repeating: 0, count: 4)

                let rLim = Int(Float(r) * 1.42)
                for dy in -rLim...rLim {
                    for dx in -rLim...rLim {
                        let px = x + dx
                        let py = y + dy
                        if px < 0 || px >= W || py < 0 || py >= H { continue }
                        let d = sqrt(Float(dx * dx + dy * dy))
                        let pIdx = (py * W + px) * 4
                        let pr = Float(pixels[pIdx])
                        let pg = Float(pixels[pIdx + 1])
                        let pb = Float(pixels[pIdx + 2])

                        if d <= Float(r) * 0.82 {
                            let q = (dy >= 0 ? 2 : 0) + (dx >= 0 ? 1 : 0)
                            qR[q] += pr; qG[q] += pg; qB[q] += pb; qN[q] += 1
                        }

                        if d <= Float(r) * 0.50 {
                            coreR += pr; coreG += pg; coreB += pb; coreCnt += 1
                        } else if d > Float(r) * 0.52 && d <= Float(r) * 0.88 {
                            rimR += pr; rimG += pg; rimB += pb; rimCnt += 1
                        } else if d >= Float(r) * 1.15 && d <= Float(r) * 1.40 {
                            outR += pr; outG += pg; outB += pb; outCnt += 1
                        }
                    }
                }

                if coreCnt < 6 || rimCnt < 6 || outCnt < 6 || qN.contains(where: { $0 < 4 }) { continue }
                for q in 0..<4 {
                    qR[q] /= qN[q]; qG[q] /= qN[q]; qB[q] /= qN[q]
                }

                let cR = coreR / coreCnt, cG = coreG / coreCnt, cB = coreB / coreCnt
                let mR = rimR / rimCnt, mG = rimG / rimCnt, mB = rimB / rimCnt
                let oR = outR / outCnt, oG = outG / outCnt, oB = outB / outCnt

                let avgR = 0.5 * (cR + mR)
                let avgG = 0.5 * (cG + mG)
                let avgB = 0.5 * (cB + mB)

                // 4.2 Simetria com tolerância para relevo de rostos/efígies (até 52.0)
                var maxQuadDiff: Float = 0
                for q in 0..<4 {
                    let diff = hypot(qR[q] - avgR, hypot(qG[q] - avgG, qB[q] - avgB))
                    if diff > maxQuadDiff { maxQuadDiff = diff }
                }
                if maxQuadDiff > 52.0 { continue }

                let warmth = avgR - avgB
                let maxDiff = max(abs(avgR - avgG), max(abs(avgG - avgB), abs(avgR - avgB)))
                let blueness = avgB - max(avgR, avgG)

                // Rejeita azul e fundo sem cor metálica
                if blueness > 6.0 || warmth < -12.0 { continue }

                // 4.3 Filtro Anti-Pele Humana
                let rgDiff = avgR - avgG
                let gbDiff = avgG - avgB
                let skinRatio = rgDiff / max(gbDiff, 1.0)
                let isSkinTone = rgDiff > 24.0 && skinRatio > 1.85 && warmth > 28.0
                if isSkinTone { continue }

                let coreWarmth = cR - cB
                let rimWarmth = mR - mB

                let isGoldOrCopper = warmth > 18.0 && avgR > 68.0 && skinRatio <= 1.85
                let isSilver = maxDiff < 20.0 && avgR > 88.0 && avgR < 236.0
                let isBimetallic = rimWarmth > 18.0 && (rimWarmth - coreWarmth) > 6.0

                guard isGoldOrCopper || isSilver || isBimetallic else { continue }

                let colorContrast = sqrt(pow(avgR - oR, 2) + pow(avgG - oG, 2) + pow(avgB - oB, 2))
                if colorContrast < 14.0 { continue }

                candidates.append(CircleCandidate(
                    x: x, y: y, r: r, score: s,
                    coreR: cR, coreG: cG, coreB: cB,
                    rimR: mR, rimG: mG, rimB: mB,
                    warmth: warmth, maxDiff: maxDiff
                ))
            }
        }

        // Ordena por score circular e permite moedas encostadas (limiar de 0.58 da soma dos raios)
        candidates.sort { ($0.score * Float($0.r)) > ($1.score * Float($1.r)) }
        var filtered: [CircleCandidate] = []
        for cand in candidates {
            let overlaps = filtered.contains { existing in
                let d = hypot(Float(existing.x - cand.x), Float(existing.y - cand.y))
                return d < Float(existing.r + cand.r) * 0.58
            }
            if !overlaps {
                filtered.append(cand)
                if filtered.count >= 55 { break }
            }
        }

        if filtered.isEmpty { return [] }

        // Identifica a distribuição dos raios das moedas douradas na cena para classificação invariante à face (cara/coroa)
        let goldRadii = filtered.filter { $0.warmth > 18.0 }.map { Float($0.r) }.sorted()
        let minGoldRadius = goldRadii.first ?? 28.0
        let maxGoldRadius = goldRadii.last ?? 38.0
        let hasBoth10And25 = (maxGoldRadius / max(minGoldRadius, 1.0)) >= 1.15

        let sortedRadii = filtered.map { Float($0.r) }.sorted()
        let medianRadius = sortedRadii[sortedRadii.count / 2]
        let scaleX = Double(cgImage.width) / Double(W)
        let scaleY = Double(cgImage.height) / Double(H)

        var outputCoins: [DetectedCoin] = []
        for cand in filtered {
            let radiusRatio = Float(cand.r) / max(medianRadius, 12.0)
            let relativeToMinGold = Float(cand.r) / max(minGoldRadius, 1.0)

            // Recorta a moeda da imagem original e passa pelo modelo CoreML (CoinClassifier)
            let pad = Double(cand.r) * 1.15
            let cropRect = CGRect(
                x: max(0.0, (Double(cand.x) - pad) * scaleX),
                y: max(0.0, (Double(cand.y) - pad) * scaleY),
                width: min(Double(cgImage.width), pad * 2.0 * scaleX),
                height: min(Double(cgImage.height), pad * 2.0 * scaleY)
            )

            var mlPrediction: CoinDenomination? = nil
            var mlConfidence: Float = 0.0
            var isBackgroundML = false

            if let vnModel = vnCoinModel, let croppedCG = cgImage.cropping(to: cropRect) {
                let request = VNCoreMLRequest(model: vnModel)
                request.imageCropAndScaleOption = .scaleFill
                let handler = VNImageRequestHandler(cgImage: croppedCG, options: [:])
                try? handler.perform([request])
                if let observations = request.results as? [VNClassificationObservation],
                   let top = observations.first {
                    let bgProb = observations.first(where: { $0.identifier == "fundo" })?.confidence ?? 0.0
                    if top.identifier == "fundo" || bgProb > 0.40 || top.confidence < 0.65 {
                        isBackgroundML = true
                    } else if let denom = CoinDenomination.fromMLIdentifier(top.identifier) {
                        mlPrediction = denom
                        mlConfidence = top.confidence
                    } else {
                        isBackgroundML = true
                    }
                } else {
                    isBackgroundML = true
                }
            }

            if isBackgroundML { continue }

            let finalDecision = combineMLAndPhysicalFeatures(
                mlPrediction: mlPrediction,
                mlConfidence: mlConfidence,
                candidate: cand,
                radiusRatio: radiusRatio,
                relativeToMinGold: relativeToMinGold,
                hasBoth10And25: hasBoth10And25,
                sceneCount: filtered.count
            )

            let normBox = CGRect(
                x: CGFloat(cand.x - cand.r) / CGFloat(W),
                y: CGFloat(cand.y - cand.r) / CGFloat(H),
                width: CGFloat(cand.r * 2) / CGFloat(W),
                height: CGFloat(cand.r * 2) / CGFloat(H)
            )

            outputCoins.append(DetectedCoin(
                denomination: finalDecision.0,
                boundingBox: normBox,
                confidence: finalDecision.1,
                center: CGPoint(x: normBox.midX, y: normBox.midY)
            ))
        }

        return outputCoins
    }

    private func combineMLAndPhysicalFeatures(
        mlPrediction: CoinDenomination?,
        mlConfidence: Float,
        candidate: CircleCandidate,
        radiusRatio: Float,
        relativeToMinGold: Float,
        hasBoth10And25: Bool,
        sceneCount: Int
    ) -> (CoinDenomination, Float) {
        let coreR = candidate.coreR, coreG = candidate.coreG, coreB = candidate.coreB
        let rimR = candidate.rimR, rimG = candidate.rimG, rimB = candidate.rimB

        // 1. Prioridade para amostras calibradas manualmente pelo usuário no dispositivo
        if !learnedSamples.isEmpty {
            var bestDist: Float = .greatestFiniteMagnitude
            var bestDenom: CoinDenomination? = nil
            for sample in learnedSamples {
                let dCore = hypot(coreR - sample.coreR, hypot(coreG - sample.coreG, coreB - sample.coreB))
                let dRim = hypot(rimR - sample.rimR, hypot(rimG - sample.rimG, rimB - sample.rimB))
                let dRad = abs(radiusRatio - sample.radiusRatio) * 60.0
                let totalDist = dCore * 0.5 + dRim * 0.35 + dRad * 0.15
                if totalDist < bestDist {
                    bestDist = totalDist
                    bestDenom = CoinDenomination(rawValue: sample.denominationRaw)
                }
            }
            if let matched = bestDenom, bestDist < 28.0 {
                return (matched, 0.96)
            }
        }

        let coreWarmth = coreR - coreB
        let rimWarmth = rimR - rimB
        let rimCoreWarmthDiff = rimWarmth - coreWarmth
        let coreMaxDiff = max(abs(coreR - coreG), max(abs(coreG - coreB), abs(coreR - coreB)))
        let rimMaxDiff = max(abs(rimR - rimG), max(abs(rimG - rimB), abs(rimR - rimB)))

        // 2. Verificação física de Bimetálica (R$ 1,00: anel externo dourado de 27mm + núcleo interno prateado de 18mm)
        if mlPrediction == .oneReal && mlConfidence > 0.65 {
            return (.oneReal, mlConfidence)
        }
        if (rimWarmth > 18.0 && rimCoreWarmthDiff >= 8.0) || (rimMaxDiff - coreMaxDiff >= 8.0 && rimWarmth > 18.0) {
            return (.oneReal, 0.94)
        }

        // 3. Moedas prateadas (Aço inoxidável: 50 centavos de 23mm ou 10 centavos de 1994)
        if candidate.maxDiff < 20.0 && candidate.warmth < 16.0 {
            if let ml = mlPrediction, (ml == .fiftyCents || ml == .tenCents || ml == .twentyFiveCents), mlConfidence > 0.65 {
                return (ml, mlConfidence)
            }
            return radiusRatio >= 1.05 ? (.fiftyCents, 0.90) : (.tenCents, 0.88)
        }

        // 4. Moeda de 5 centavos (Aço revestido de cobre avermelhado intenso de 22mm)
        let avgR = 0.5 * (coreR + rimR)
        let avgG = 0.5 * (coreG + rimG)
        let avgB = 0.5 * (coreB + rimB)
        let greenRatio = (avgG - avgB) / max(avgR - avgB, 1.0)
        if greenRatio < 0.50 && candidate.warmth > 26.0 {
            if mlPrediction == .fiveCents {
                return (.fiveCents, max(mlConfidence, 0.93))
            }
            return (.fiveCents, 0.89)
        }

        // 5. Moedas Douradas (10 Centavos = 20mm vs 25 Centavos = 25mm - 25% maior em raio!)
        // Invariante ao fato de estar virada para cima (número) ou virada para baixo (rosto/efígie)
        if hasBoth10And25 {
            if relativeToMinGold >= 1.15 {
                return (.twentyFiveCents, max(mlConfidence, 0.94))
            } else {
                return (.tenCents, max(mlConfidence, 0.94))
            }
        }

        // Se a cena tem múltiplas moedas com razão relativa à mediana
        if sceneCount >= 3 {
            if radiusRatio >= 1.10 {
                return (.twentyFiveCents, max(mlConfidence, 0.92))
            } else if radiusRatio <= 0.98 {
                return (.tenCents, max(mlConfidence, 0.92))
            }
        }

        // Caso contrário, usa a decisão da IA Local CoreML treinada
        if let ml = mlPrediction {
            return (ml, max(mlConfidence, 0.85))
        }

        return relativeToMinGold >= 1.14 ? (.twentyFiveCents, 0.85) : (.tenCents, 0.85)
    }

    private func extractCenterCoinSample(from pixelBuffer: CVPixelBuffer, target: CoinDenomination) -> LearnedCoinSample? {
        guard let cgImage = createCGImage(from: pixelBuffer) else { return nil }
        let W = 256
        let H = 256
        var pixels = [UInt8](repeating: 0, count: W * H * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: &pixels,
            width: W,
            height: H,
            bitsPerComponent: 8,
            bytesPerRow: W * 4,
            space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: W, height: H))

        let cx = W / 2
        let cy = H / 2
        let r = 24
        var cR: Float = 0, cG: Float = 0, cB: Float = 0, cN: Float = 0
        var mR: Float = 0, mG: Float = 0, mB: Float = 0, mN: Float = 0

        for dy in -r...r {
            for dx in -r...r {
                let d = sqrt(Float(dx * dx + dy * dy))
                let idx = ((cy + dy) * W + (cx + dx)) * 4
                let pr = Float(pixels[idx])
                let pg = Float(pixels[idx + 1])
                let pb = Float(pixels[idx + 2])
                if d <= Float(r) * 0.5 {
                    cR += pr; cG += pg; cB += pb; cN += 1
                } else if d <= Float(r) * 0.9 {
                    mR += pr; mG += pg; mB += pb; mN += 1
                }
            }
        }
        guard cN > 0 && mN > 0 else { return nil }
        cR /= cN; cG /= cN; cB /= cN
        mR /= mN; mG /= mN; mB /= mN

        // Impede calibrar no fundo azul
        if (cB - max(cR, cG)) > 10.0 { return nil }

        return LearnedCoinSample(
            denominationRaw: target.rawValue,
            coreR: cR, coreG: cG, coreB: cB,
            rimR: mR, rimG: mG, rimB: mB,
            radiusRatio: 1.0
        )
    }

    private func createCGImage(from pixelBuffer: CVPixelBuffer) -> CGImage? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext(options: [.useSoftwareRenderer: false])
        return context.createCGImage(ciImage, from: ciImage.extent)
    }
}

// MARK: - Preview da Câmera
struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {}

    final class PreviewUIView: UIView {
        override class var layerClass: AnyClass {
            AVCaptureVideoPreviewLayer.self
        }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer {
            layer as! AVCaptureVideoPreviewLayer
        }
    }
}

// MARK: - Interface Principal Responsiva e Moderna
struct ContentView: View {
    @StateObject private var engine = CoinVisionEngine()
    @State private var showCalibrationSheet = false

    var body: some View {
        ZStack {
            Color(red: 0.05, green: 0.06, blue: 0.08)
                .ignoresSafeArea()

            VStack(spacing: 8) {
                // 1. Cabeçalho Responsivo com Safe Area Respeitada
                headerBar

                // 2. Visor da Câmera / Foto com Proporção Equilibrada
                ZStack {
                    if let snapshot = engine.capturedSnapshot {
                        Image(uiImage: snapshot)
                            .resizable()
                            .scaledToFill()
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .clipped()
                    } else {
                        CameraPreviewView(session: engine.session)
                    }

                    // Overlay das moedas detectadas pela IA Local
                    GeometryReader { geo in
                        ForEach(engine.detectedCoins) { coin in
                            let rect = CGRect(
                                x: coin.boundingBox.origin.x * geo.size.width,
                                y: coin.boundingBox.origin.y * geo.size.height,
                                width: coin.boundingBox.width * geo.size.width,
                                height: coin.boundingBox.height * geo.size.height
                            )

                            ZStack {
                                Circle()
                                    .stroke(coin.denomination.ringColor, lineWidth: 3)
                                    .background(Circle().fill(coin.denomination.badgeColor.opacity(0.22)))
                                    .frame(width: max(rect.width, 32), height: max(rect.height, 32))

                                Text(coin.denomination.label)
                                    .font(.system(size: 11, weight: .black, design: .rounded))
                                    .foregroundColor(.black)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(
                                        Capsule()
                                            .fill(coin.denomination.badgeColor)
                                            .shadow(color: .black.opacity(0.6), radius: 2)
                                    )
                            }
                            .position(x: rect.midX, y: rect.midY)
                        }

                        // Mira central para o Modo Manual
                        if engine.capturedSnapshot == nil && !engine.isProcessingPhoto {
                            Circle()
                                .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [6, 5]))
                                .foregroundColor(.white.opacity(0.40))
                                .frame(width: 86, height: 86)
                                .position(x: geo.size.width / 2, y: geo.size.height / 2)
                        }
                    }

                    // Overlay de Loading em Tempo Real durante a análise da foto pela IA Local
                    if engine.isProcessingPhoto {
                        ZStack {
                            Color.black.opacity(0.55)
                            VStack(spacing: 10) {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .yellow))
                                    .scaleEffect(1.3)
                                Text("Analisando moedas com IA Local...")
                                    .font(.system(size: 13, weight: .bold, design: .rounded))
                                    .foregroundColor(.white)
                                Text("Verificando tamanho oficial, faces e CoreML")
                                    .font(.system(size: 11, weight: .medium, design: .rounded))
                                    .foregroundColor(.yellow.opacity(0.9))
                            }
                            .padding(.horizontal, 20)
                            .padding(.vertical, 14)
                            .background(Color.black.opacity(0.85))
                            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                            .overlay(
                                RoundedRectangle(cornerRadius: 16, style: .continuous)
                                    .stroke(Color.yellow.opacity(0.5), lineWidth: 1.5)
                            )
                        }
                    }

                    // Barra superior dentro do visor
                    VStack {
                        HStack {
                            HStack(spacing: 5) {
                                Circle()
                                    .fill(engine.isAutoCountingSweep ? Color.green : Color.yellow)
                                    .frame(width: 7, height: 7)
                                Text(engine.capturedSnapshot != nil ? "FOTO ANALISADA (IA)" : (engine.isAutoCountingSweep ? "TEMPO REAL" : "IA COREML"))
                                    .font(.system(size: 10, weight: .heavy, design: .rounded))
                                    .foregroundColor(.white)
                            }
                            .padding(.horizontal, 9)
                            .padding(.vertical, 5)
                            .background(Color.black.opacity(0.65))
                            .clipShape(Capsule())

                            Spacer()

                            if engine.capturedSnapshot != nil {
                                Button(action: { engine.dismissSnapshot() }) {
                                    HStack(spacing: 4) {
                                        Image(systemName: "video.fill")
                                        Text("Voltar p/ Câmera")
                                    }
                                    .font(.system(size: 10, weight: .bold, design: .rounded))
                                    .foregroundColor(.black)
                                    .padding(.horizontal, 9)
                                    .padding(.vertical, 5)
                                    .background(Color.yellow)
                                    .clipShape(Capsule())
                                }
                            } else {
                                Button(action: { showCalibrationSheet = true }) {
                                    HStack(spacing: 4) {
                                        Image(systemName: "cpu")
                                        Text("Calibrar IA")
                                    }
                                    .font(.system(size: 10, weight: .bold, design: .rounded))
                                    .foregroundColor(.cyan)
                                    .padding(.horizontal, 9)
                                    .padding(.vertical, 5)
                                    .background(Color.black.opacity(0.65))
                                    .clipShape(Capsule())
                                }
                            }
                        }
                        .padding(8)

                        Spacer()

                        Text(engine.statusMessage)
                            .font(.system(size: 11, weight: .semibold, design: .rounded))
                            .foregroundColor(.white)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(Color.black.opacity(0.75))
                            .clipShape(Capsule())
                            .padding(.bottom, 6)
                    }
                }
                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .stroke(engine.isAutoCountingSweep ? Color.green : Color.yellow.opacity(0.40), lineWidth: 1.5)
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                // 3. Os 3 Botões Principais Solicitados
                threeModeButtons

                // 4. Grade de Moedas (5¢, 10¢, 25¢, 50¢, R$ 1)
                coinCountersStrip
            }
            .padding(.horizontal, 10)
            .padding(.top, 4)
            .padding(.bottom, 4)
        }
        .sheet(isPresented: $showCalibrationSheet) {
            calibrationSheet
        }
    }

    private var headerBar: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 1) {
                Text("TOTAL ACUMULADO")
                    .font(.system(size: 9, weight: .bold, design: .rounded))
                    .foregroundColor(.green)
                Text(formatCurrency(engine.totalReais))
                    .font(.system(size: 22, weight: .black, design: .rounded))
                    .foregroundColor(.white)
                    .minimumScaleFactor(0.7)
                    .lineLimit(1)
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 1) {
                Text("NA TELA (\(engine.detectedCoins.count))")
                    .font(.system(size: 9, weight: .bold, design: .rounded))
                    .foregroundColor(.yellow)
                Text(formatCurrency(engine.visibleTotalReais))
                    .font(.system(size: 15, weight: .heavy, design: .rounded))
                    .foregroundColor(.yellow)
            }
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(Color.yellow.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 10))

            Button(action: { engine.toggleTorch() }) {
                Image(systemName: engine.torchOn ? "flashlight.on.fill" : "flashlight.off.fill")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundColor(engine.torchOn ? .black : .white)
                    .frame(width: 34, height: 34)
                    .background(engine.torchOn ? Color.yellow : Color.white.opacity(0.12))
                    .clipShape(Circle())
            }

            Button(action: { engine.resetAll() }) {
                Image(systemName: "arrow.counterclockwise")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundColor(.white)
                    .frame(width: 34, height: 34)
                    .background(Color.red.opacity(0.85))
                    .clipShape(Circle())
            }
        }
        .padding(.horizontal, 4)
    }

    private var threeModeButtons: some View {
        HStack(spacing: 6) {
            // BOTÃO 1: Tirar Foto e Identificar com IA Local
            Button(action: { engine.capturePhotoAndIdentify() }) {
                VStack(spacing: 2) {
                    Image(systemName: "camera.viewfinder")
                        .font(.system(size: 18, weight: .bold))
                    Text("TIRAR FOTO (IA)")
                        .font(.system(size: 10, weight: .black, design: .rounded))
                    Text("Fotografa e soma")
                        .font(.system(size: 8, weight: .medium, design: .rounded))
                        .opacity(0.85)
                }
                .foregroundColor(.black)
                .frame(maxWidth: .infinity)
                .frame(height: 50)
                .background(
                    LinearGradient(
                        colors: [Color.yellow, Color.orange],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            }

            // BOTÃO 2: Em Tempo Real (Contínuo)
            Button(action: { engine.toggleRealtimeMode() }) {
                VStack(spacing: 2) {
                    Image(systemName: engine.isAutoCountingSweep ? "pause.circle.fill" : "dot.radiowaves.left.and.right")
                        .font(.system(size: 18, weight: .bold))
                    Text("TEMPO REAL")
                        .font(.system(size: 10, weight: .black, design: .rounded))
                    Text(engine.isAutoCountingSweep ? "Contando..." : "Passar câmera")
                        .font(.system(size: 8, weight: .medium, design: .rounded))
                        .opacity(0.85)
                }
                .foregroundColor(engine.isAutoCountingSweep ? .black : .white)
                .frame(maxWidth: .infinity)
                .frame(height: 50)
                .background(
                    engine.isAutoCountingSweep
                    ? AnyShapeStyle(Color.green)
                    : AnyShapeStyle(Color.white.opacity(0.14))
                )
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            }

            // BOTÃO 3: Manual (1 clique por vez)
            Button(action: { engine.scanSingleCoinManual() }) {
                VStack(spacing: 2) {
                    Image(systemName: "plus.viewfinder")
                        .font(.system(size: 18, weight: .bold))
                    Text("MANUAL (1x)")
                        .font(.system(size: 10, weight: .black, design: .rounded))
                    Text("1 clique por vez")
                        .font(.system(size: 8, weight: .medium, design: .rounded))
                        .opacity(0.85)
                }
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .frame(height: 50)
                .background(Color.blue.opacity(0.85))
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            }
        }
    }

    private var coinCountersStrip: some View {
        HStack(spacing: 5) {
            ForEach(CoinDenomination.allCases) { coin in
                let count = engine.counts[coin, default: 0]
                VStack(spacing: 2) {
                    Button(action: { engine.incrementManual(coin) }) {
                        ZStack {
                            Circle()
                                .fill(coin.badgeColor)
                                .frame(width: 34, height: 34)
                            if coin == .oneReal {
                                Circle()
                                    .fill(Color(white: 0.85))
                                    .frame(width: 20, height: 20)
                            }
                            Text(coin.label)
                                .font(.system(size: 10, weight: .black, design: .rounded))
                                .foregroundColor(.black)
                        }
                    }

                    Text("x\(count)")
                        .font(.system(size: 12, weight: .heavy, design: .rounded))
                        .foregroundColor(count > 0 ? .white : .gray)

                    Button(action: { engine.decrementManual(coin) }) {
                        Image(systemName: "minus.circle.fill")
                            .font(.system(size: 14))
                            .foregroundColor(count > 0 ? .white.opacity(0.7) : .white.opacity(0.2))
                    }
                    .disabled(count == 0)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 4)
                .background(Color.white.opacity(0.06))
                .clipShape(RoundedRectangle(cornerRadius: 10))
            }
        }
    }

    private var calibrationSheet: some View {
        NavigationStack {
            VStack(spacing: 14) {
                Text("Centralize uma moeda na mira da câmera e toque no valor correspondente para adicionar uma amostra extra de iluminação à IA Local.")
                    .font(.system(size: 14, weight: .medium, design: .rounded))
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)

                ForEach(CoinDenomination.allCases) { coin in
                    Button(action: {
                        engine.learnCoinFromCamera(target: coin)
                        showCalibrationSheet = false
                    }) {
                        HStack {
                            Circle()
                                .fill(coin.badgeColor)
                                .frame(width: 28, height: 28)
                            Text("Amostra de \(coin.fullName) (\(coin.label))")
                                .font(.system(size: 16, weight: .bold, design: .rounded))
                            Spacer()
                            Image(systemName: "sparkles")
                        }
                        .foregroundColor(.white)
                        .padding()
                        .background(Color.white.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                }

                if engine.learnedSamplesCount > 0 {
                    Button(role: .destructive, action: {
                        engine.clearLearnedSamples()
                        showCalibrationSheet = false
                    }) {
                        Text("Restaurar Padrão da IA CoreML")
                            .font(.system(size: 14, weight: .bold))
                    }
                    .padding(.top, 8)
                }

                Spacer()
            }
            .padding()
            .navigationTitle("Calibrar IA Local")
            .navigationBarTitleDisplayMode(.inline)
        }
        .presentationDetents([.medium])
    }

    private func formatCurrency(_ value: Double) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.locale = Locale(identifier: "pt_BR")
        return formatter.string(from: NSNumber(value: value)) ?? String(format: "R$ %.2f", value)
    }
}
