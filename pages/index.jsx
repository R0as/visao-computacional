"use client";

import React, { useEffect, useRef, useState } from "react";
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';

export default function DetectorPage() {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const rafRef = useRef(null);

    const cocoRef = useRef(null);
    const graphModelRef = useRef(null);
    const mobilenetRef = useRef(null);
    const trainingDataRef = useRef([]);
    const classLabelsRef = useRef([]);
    const lastPredictionTextRef = useRef('');

    const [isCameraOn, setIsCameraOn] = useState(false);
    const [loadingModel, setLoadingModel] = useState(false);
    const [status, setStatus] = useState("Nenhum modelo carregado");
    const [useCoco, setUseCoco] = useState(true);
    const [customModelUrl, setCustomModelUrl] = useState("");
    const [minScore, setMinScore] = useState(0.5);
    const [fps, setFps] = useState(0);
    const [useKnn, setUseKnn] = useState(false);
    const [k, setK] = useState(3);
    const [exampleLabel, setExampleLabel] = useState("");
    const [labelCounts, setLabelCounts] = useState({});

    function updateTrainingData(newData) {
        trainingDataRef.current = newData;
        const counts = {};
        newData.forEach(s => counts[s.label] = (counts[s.label] || 0) + 1);
        setLabelCounts(counts);
    }

    async function startCamera() {
    try {
        const constraints = {
            audio: false,
            video: {
                width: 640,
                height: 480,
                facingMode: { ideal: 'environment' } // <-- A MUDANÇA ESTÁ AQUI
            }
        };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        
        if (videoRef.current) {
            videoRef.current.srcObject = stream;
            await videoRef.current.play();
            setIsCameraOn(true);
            setStatus("Câmera ligada");
        }
    } catch (err) {
        console.error(err);
        setStatus("Erro ao acessar a câmera: " + err.message);
    }
}

    function stopCamera() {
        if (rafRef.current) cancelAnimationFrame(rafRef.current);
        const stream = videoRef.current?.srcObject;
        if (stream) {
            const tracks = stream.getTracks();
            tracks.forEach((t) => t.stop());
            videoRef.current.srcObject = null;
        }
        setIsCameraOn(false);
        setStatus("Câmera desligada");
    }

    function handleKnnToggle() {
    const isActivating = !useKnn;
    setUseKnn(isActivating);

    // Se estivermos ATIVANDO o k-NN, devemos desativar os outros modos.
    if (isActivating) {
        setUseCoco(false);
    }
}

    async function loadCocoModel() {
        setLoadingModel(true);
        setStatus("Carregando Coco-SSD...");
        try {
            const cocoSsd = await import("@tensorflow-models/coco-ssd");
            cocoRef.current = await cocoSsd.load();
            setUseCoco(true);
            setUseKnn(false);
            graphModelRef.current = null;
            classLabelsRef.current = [];
            setStatus("Coco-SSD carregado");
        } catch (err) {
            console.error(err);
            setStatus("Erro ao carregar Coco-SSD: " + err.message);
        } finally {
            setLoadingModel(false);
        }
    }

    async function loadCustomModel(url) {
        if (!url) {
            setStatus("Por favor, insira a URL do modelo.");
            return;
        }
        setLoadingModel(true);
        setStatus("Carregando modelo customizado...");
        try {
            const tf = await import("@tensorflow/tfjs");
            graphModelRef.current = await tf.loadLayersModel(url);
            const metadataUrl = url.replace('model.json', 'metadata.json');
            const response = await fetch(metadataUrl);
            const metadata = await response.json();
            classLabelsRef.current = metadata.labels;
            setUseCoco(false);
            setUseKnn(false);
            setStatus(`Modelo customizado e ${classLabelsRef.current.length} rótulos carregados.`);
        } catch (err) {
            console.error(err);
            setStatus("Erro ao carregar modelo customizado: " + err.message);
        } finally {
            setLoadingModel(false);
        }
    }

    async function loadMobileNet() {
    // Correção: Adiciona uma mensagem de status se o modelo já estiver carregado
    if (mobilenetRef.current) {
        setStatus("MobileNet já está carregado.");
        return mobilenetRef.current;
    }

    setLoadingModel(true);
    setStatus("Carregando MobileNet (feature extractor)...");
    try {
        const mobilenetModule = await import("@tensorflow-models/mobilenet");
        mobilenetRef.current = await mobilenetModule.load();
        setStatus("MobileNet carregado");
        return mobilenetRef.current;
    } catch (err) {
        console.error(err);
        setStatus("Erro ao carregar MobileNet: " + err.message);
    } finally {
        setLoadingModel(false);
    }
}

    function euclideanDistance(a, b) {
        let sum = 0;
        for (let i = 0; i < a.length; i++) {
            const d = a[i] - b[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    function knnClassify(inputFeature, kNearest = 3) {
    const training = trainingDataRef.current;
    if (!training || training.length === 0) return [];

    const distances = training.map(sample => ({ label: sample.label, dist: euclideanDistance(inputFeature, sample.features) }));
    distances.sort((a, b) => a.dist - b.dist);
    const topK = distances.slice(0, kNearest);
    
    const counts = {};
    topK.forEach(n => counts[n.label] = (counts[n.label] || 0) + 1);

    const results = Object.entries(counts)
        .map(([label, count]) => ({
            label: label,
            confidence: count / kNearest // Calcula a % de votos
        }))
        .sort((a, b) => b.confidence - a.confidence); // Ordena pelo mais votado

    return results; // Retorna a lista completa de resultados
}

    async function addExample(label) {
    if (!label) {
        setStatus("Informe um rótulo antes de adicionar exemplos.");
        return;
    }
    await loadMobileNet();
    const mobilenet = mobilenetRef.current;
    if (!mobilenet || !videoRef.current) {
        setStatus("MobileNet ou câmera não disponíveis.");
        return;
    }
    try {
        const tf = await import("@tensorflow/tfjs");
        const features = tf.tidy(() => {
            const activation = mobilenet.infer(videoRef.current, true);
            return Array.from(activation.dataSync());
        });
        const newTrainingData = [...trainingDataRef.current, { label, features }];
        updateTrainingData(newTrainingData);
        setStatus(`Exemplo adicionado para: ${label} (total: ${newTrainingData.length})`);
        localStorage.setItem('knnDataset', JSON.stringify(newTrainingData));
        // A linha setExampleLabel(""); foi removida daqui.
    } catch (err) {
        console.error(err);
        setStatus("Erro ao capturar exemplo: " + err.message);
    }
}

    function saveDataset() {
        try {
            localStorage.setItem('knnDataset', JSON.stringify(trainingDataRef.current));
            setStatus('Dataset salvo no localStorage.');
        } catch (err) {
            console.error(err);
            setStatus('Erro ao salvar dataset: ' + err.message);
        }
    }

    function loadDataset() {
        try {
            const raw = localStorage.getItem('knnDataset');
            if (!raw) {
                setStatus('Nenhum dataset encontrado no localStorage.');
                return;
            }
            const parsed = JSON.parse(raw);
            updateTrainingData(parsed);
            setStatus(`Dataset carregado (${parsed.length} exemplos).`);
        } catch (err) {
            console.error(err);
            setStatus('Erro ao carregar dataset: ' + err.message);
        }
    }

    function clearDataset() {
        updateTrainingData([]);
        localStorage.removeItem('knnDataset');
        setStatus('Dataset limpo.');
    }

    async function predictKNN() {
        if (!mobilenetRef.current) await loadMobileNet();
        const mobilenet = mobilenetRef.current;
        if (!mobilenet || trainingDataRef.current.length === 0) return { label: null, confidence: 0 };
        const tf = await import("@tensorflow/tfjs");
        const features = tf.tidy(() => {
            const activation = mobilenet.infer(videoRef.current, true);
            return Array.from(activation.dataSync());
        });
        const out = knnClassify(features, Math.min(k, trainingDataRef.current.length));
        return out;
    }

    function drawCocoDetections(ctx, predictions) {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        ctx.drawImage(videoRef.current, 0, 0, ctx.canvas.width, ctx.canvas.height);
        predictions.forEach(pred => {
            if (pred.score < minScore) return;
            const [x, y, w, h] = pred.bbox;
            ctx.strokeStyle = '#00FFFF';
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);
            const text = `${pred.class} ${(pred.score * 100).toFixed(1)}%`;
            ctx.font = '16px Arial';
            ctx.fillStyle = '#00FFFF';
            const textWidth = ctx.measureText(text).width;
            ctx.fillRect(x, y > 20 ? y - 20 : y, textWidth + 6, 22);
            ctx.fillStyle = '#000';
            ctx.fillText(text, x + 3, y > 20 ? y - 5 : y + 15);
        });
    }

    useEffect(() => {
        const offscreenCanvas = document.createElement('canvas');

        const predictionInterval = setInterval(async () => {
    if (isCameraOn && graphModelRef.current && !useCoco && !useKnn) {
        const video = videoRef.current;
        if (!video || video.readyState < 2 || video.videoWidth === 0) return;
        try {
            // ... (a parte de preparação do offscreenCanvas e da predição continua igual)
            offscreenCanvas.width = video.videoWidth;
            offscreenCanvas.height = video.videoHeight;
            offscreenCanvas.getContext('2d').drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
            
            const tf = await import("@tensorflow/tfjs");
            const result = tf.tidy(() => {
                const imgTensor = tf.browser.fromPixels(offscreenCanvas).toFloat();
                const inputShape = graphModelRef.current.inputs[0].shape;
                let resizedTensor = imgTensor;
                if (inputShape && inputShape.length === 4) {
                    const targetH = inputShape[1] || video.videoHeight;
                    const targetW = inputShape[2] || video.videoWidth;
                    resizedTensor = tf.image.resizeBilinear(imgTensor, [targetH, targetW]);
                }
                const normalized = resizedTensor.div(255.0);
                const batched = normalized.expandDims(0);
                return graphModelRef.current.predict(batched);
            });

            const scoresTensor = Array.isArray(result) ? result[result.length - 1] : result;
            const scores = await scoresTensor.data();
            const labels = classLabelsRef.current;

            // --- MUDANÇA PRINCIPAL AQUI ---
            // Em vez de pegar só o maior, vamos criar uma lista com todos.
            const allPredictions = Array.from(scores)
                .map((score, index) => ({
                    label: labels[index] || `Classe ${index}`,
                    confidence: score
                }))
                .sort((a, b) => b.confidence - a.confidence) // Ordena da maior confiança para a menor
                .filter(p => p.confidence > 0.01); // Filtra resultados muito pequenos para não poluir a tela

            // Formata a lista para exibição e a salva na nossa ref
            lastPredictionTextRef.current = allPredictions; 
            
            // Limpeza de memória
            if (result) {
                if (Array.isArray(result)) result.forEach(r => r.dispose()); else result.dispose();
            }
            if (scoresTensor && scoresTensor !== result) scoresTensor.dispose();

        } catch (err) {
            console.error("Erro na predição em background:", err);
            lastPredictionTextRef.current = [{ label: "Erro na predição", confidence: 1 }];
        }
    } else {
        lastPredictionTextRef.current = [];
    }
}, 100);

        let rafId;
        async function frameLoop() {
            const video = videoRef.current;
            const canvas = canvasRef.current;
            if (!isCameraOn || !video || !canvas || video.readyState < 2 || video.videoWidth === 0) {
                rafId = requestAnimationFrame(frameLoop);
                return;
            }
            const ctx = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            if (useCoco && cocoRef.current) {
                const predictions = await cocoRef.current.detect(video);
                drawCocoDetections(ctx, predictions);
// Dentro do frameLoop, substitua o bloco 'else if (useKnn)'

} else if (useKnn) {
    const predictions = await predictKNN(); // Agora predictKNN retorna um array

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (predictions && predictions.length > 0) {
        predictions.forEach((prediction, index) => {
            const text = `${prediction.label} — ${(prediction.confidence * 100).toFixed(0)}%`;
            const yPos = canvas.height - 40 - (index * 30);

            ctx.font = '20px Arial';
            const textWidth = ctx.measureText(text).width;
            
            ctx.fillStyle = 'rgba(0,0,0,0.6)';
            ctx.fillRect(8, yPos - 22, textWidth + 12, 28);

            ctx.fillStyle = index === 0 ? '#00FFAA' : '#FFFFFF';

            ctx.fillText(text, 12, yPos);
        });
    }
} else {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    const predictionsToDraw = lastPredictionTextRef.current;

    // Verifica se há predições para desenhar
    if (predictionsToDraw && predictionsToDraw.length > 0) {
        
        // Desenha cada predição em uma linha
        predictionsToDraw.forEach((prediction, index) => {
            const text = `${prediction.label} — ${(prediction.confidence * 100).toFixed(1)}%`;
            const yPos = canvas.height - 40 - (index * 30); // Calcula a posição Y para cada linha

            ctx.font = '20px Arial';
            const textWidth = ctx.measureText(text).width;

            ctx.fillStyle = 'rgba(0,0,0,0.6)';
            ctx.fillRect(8, yPos - 22, textWidth + 12, 28);
            
            // Destaca a predição principal
            ctx.fillStyle = index === 0 ? '#00FFAA' : '#FFFFFF'; 
            
            ctx.fillText(text, 12, yPos);
        });
    }
}
            rafId = requestAnimationFrame(frameLoop);
        }

        if (isCameraOn) {
            frameLoop();
        }

        return () => {
            clearInterval(predictionInterval);
            if (rafId) cancelAnimationFrame(rafId);
        };
    }, [isCameraOn, useCoco, useKnn]);

    return (
        <div className="min-h-screen flex flex-col items-center bg-gray-50 p-6 text-black">
            <h1 className="text-2xl text-black font-bold mb-4">Visão Computacional — Detector + k-NN (treinado por você)</h1>
            <div className="w-full max-w-4xl bg-white rounded-lg shadow p-4 space-y-4">
                <div className="flex gap-2 flex-wrap">
                    <button className="px-3 py-2 rounded bg-blue-600 text-white" onClick={startCamera} disabled={isCameraOn}>Ligar Câmera</button>
                    <button className="px-3 py-2 rounded bg-red-600 text-white" onClick={stopCamera} disabled={!isCameraOn}>Desligar Câmera</button>
                    <button className="px-3 py-2 rounded bg-green-600 text-white" onClick={loadCocoModel} disabled={loadingModel}>Carregar Coco-SSD</button>
                    <input value={customModelUrl} onChange={(e) => setCustomModelUrl(e.target.value)} className="px-2 py-1 border rounded" placeholder="URL do model.json" />
                    <button className="px-3 py-2 rounded bg-indigo-600 text-white" onClick={() => loadCustomModel(customModelUrl)} disabled={loadingModel}>Carregar modelo customizado</button>
                    <button className="px-3 py-2 rounded bg-yellow-600 text-black" onClick={loadMobileNet} disabled={loadingModel}>Carregar MobileNet (p/ k-NN)</button>
                </div>
                <div className="flex gap-4 items-center">
                    <label className="text-black flex items-center gap-2">Confiança mínima (Coco):
                        <input type="range" min="0" max="1" step="0.01" value={minScore} onChange={(e) => setMinScore(parseFloat(e.target.value))} />
                        <span className="ml-2 font-mono">{minScore.toFixed(2)}</span>
                    </label>
                    <div className="ml-auto text-sm text-black">
                        <div>Status: <span className="font-medium">{status}</span></div>
                        <div>FPS: <span className="font-medium">{fps}</span></div>
                    </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-black flex items-center justify-center">
                        <video ref={videoRef} className="max-w-full" playsInline muted />
                    </div>
                    <div className="bg-black flex items-center justify-center">
                        <canvas ref={canvasRef} className="max-w-full" />
                    </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                        <h3 className="font-semibold text-black">Treinar k-NN (no navegador)</h3>
                        <div className="flex gap-2">
                            <input className="px-2 py-1 border rounded flex-1" placeholder="Rótulo do exemplo (ex: caneta)" value={exampleLabel} onChange={(e) => setExampleLabel(e.target.value)} />
                            <button className="px-3 py-2 rounded bg-sky-600 text-white" onClick={() => addExample(exampleLabel)}>Adicionar exemplo</button>
                        </div>
                        <div className="flex gap-2 items-center">
                            <label className="flex items-center gap-2">k:
                                <input type="number" min="1" max="20" value={k} onChange={(e) => setK(parseInt(e.target.value || 1))} className="w-16 px-2 py-1 border rounded" />
                            </label>
                            <button className={`px-3 py-2 rounded ${useKnn ? 'bg-red-600' : 'bg-green-600'} text-white`} onClick={handleKnnToggle}>{useKnn ? 'Desativar K-NN' : 'Ativar K-NN'}</button>
                        </div>
                        <div className="flex gap-2">
                            <button className="px-3 py-2 rounded bg-amber-500" onClick={saveDataset}>Salvar dataset</button>
                            <button className="px-3 py-2 rounded bg-amber-700 text-white" onClick={loadDataset}>Carregar dataset</button>
                            <button className="px-3 py-2 rounded bg-gray-400" onClick={clearDataset}>Limpar dataset</button>
                        </div>
                        <div className="text-sm text-black">
                            <p>Contagem de exemplos por rótulo:</p>
                            <pre className="bg-gray-100 p-2 rounded text-xs">{JSON.stringify(labelCounts, null, 2)}</pre>
                        </div>
                    </div>
                    <div className="text-black">
                        <h3 className="font-semibold">Modo de teste e notas</h3>
                        <ol className="list-decimal pl-6 text-sm">
                            <li>Adicione pelo menos 10-20 exemplos por rótulo para começar.</li>
                            <li>Ative o K-NN para usar seu classificador; desative para voltar ao modelo ativo.</li>
                            <li>Use Salvar/Carregar para não perder os dados.</li>
                        </ol>
                    </div>
                </div>
            </div>
        </div>
    );
}