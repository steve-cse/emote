import  { useState, useRef, useEffect, ChangeEvent } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as blazeface from '@tensorflow-models/blazeface';
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ThemeToggle } from "@/components/theme-toggle"

interface EmotionPrediction {
  emotion: string;
  probability: number;
}

const emotionEmojis: { [key: string]: string } = {
  'Angry': '😠',
  'Disgust': '🤢',
  'Fear': '😨',
  'Happy': '😊',
  'Neutral': '😐',
  'Sad': '😢',
  'Surprised': '😮'
};

function Emote() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [emotionPredictions, setEmotionPredictions] = useState<EmotionPrediction[] | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [models, setModels] = useState<{
    faceDetector: any;
    emotionDetector: any;
  }>({
    faceDetector: null,
    emotionDetector: null
  });

  // Load both models on component mount
  useEffect(() => {
    const loadModels = async () => {
      try {
        const faceModel = await blazeface.load();
        const emotionModel = await tf.loadLayersModel('model/model.json');
        
        setModels({
          faceDetector: faceModel,
          emotionDetector: emotionModel
        });
      } catch (error) {
        console.error("Error loading models:", error);
      }
    };

    loadModels();
  }, []);

  const handleImageUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e: ProgressEvent<FileReader>) => {
        if (typeof e.target?.result === 'string') {
          setSelectedImage(e.target.result);
          setEmotionPredictions(null);
          setPrediction(null);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const detectEmotion = async () => {
    if (!selectedImage || !imageRef.current || !models.faceDetector || !models.emotionDetector) return;

    try {
      setIsLoading(true);
      
      // Set up canvas
      const canvas = canvasRef.current;
      if (!canvas) return;
      
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      canvas.width = imageRef.current.width;
      canvas.height = imageRef.current.height;
      ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height);
      
      // Get image data
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      
      // Detect face
      const faces = await models.faceDetector.estimateFaces(imageData);
      
      if (faces.length > 0) {
        const face = faces[0];
        const landmarks = face.landmarks;
        const nosex = landmarks[2][0];
        const nosey = landmarks[2][1];
        const right = landmarks[4][0];
        const left = landmarks[5][0];
        const length = (left - right) / 2 + 5;

        // Crop face region
        const faceData = ctx.getImageData(nosex - length, nosey - length, 2 * length, 2 * length);
        
        // Convert to tensor and preprocess
        const imageTensor = tf.browser.fromPixels(faceData)
          .resizeBilinear([48, 48])
          .mean(2)
          .toFloat()
          .expandDims(0)
          .expandDims(-1);

        // Predict emotion
        const result = models.emotionDetector.predict(imageTensor);
        const predictions = await result.array();
        
        const emotions = [
          'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised'
        ];
        
        console.info(predictions); 

        const emotionResults: EmotionPrediction[] = predictions[0].map((prob: number, idx: number) => ({
          emotion: emotions[idx],
          probability: prob * 100
        }));

        setEmotionPredictions(emotionResults);
        setPrediction("Face detected and emotion analyzed!");
      } else {
        setPrediction("No faces detected");
        setEmotionPredictions(null);
      }
    } catch (error) {
      console.error("Error detecting faces:", error);
      setPrediction("Error detecting faces");
    } finally {
      setIsLoading(false);
    }
  };

  return (
      <div className="flex flex-col items-center min-h-screen p-8 relative">
        <div className="absolute bottom-4 right-4">
          <ThemeToggle />
        </div>
        <h1 className="scroll-m-20 text-4xl font-extrabold tracking-tight lg:text-5xl mb-8">Emotion Detector</h1>
        
        <div className="flex gap-4 justify-center items-center w-full max-w-2xl">
          <div className="flex-1">
            <Input 
              id="picture" 
              type="file" 
              accept="image/*"
              onChange={handleImageUpload}
            />
          </div>
          <Button 
            onClick={detectEmotion}
            disabled={!selectedImage || isLoading || !models.faceDetector || !models.emotionDetector}
            className="cursor-pointer"
          >
            {isLoading ? 'Processing....' : 'Detect Emotion'}
          </Button>
        </div>

        {selectedImage && (
          <div className="mt-8 flex justify-center w-full px-4">
            <img
              ref={imageRef}
              src={selectedImage}
              alt="Selected"
              className="shadow-lg w-full max-w-[500px] h-auto object-contain"
            />
            <canvas
              ref={canvasRef}
              style={{ display: 'none' }}
            />
          </div>
        )}

        {prediction && (
          <div className="prediction mt-8 text-center">
            <h2 className="text-2xl font-bold mb-4">Result:</h2>
            <p className="mb-4 leading-7 [&:not(:first-child)]:mt-6">{prediction}</p>
            {emotionPredictions && (
              <div className="emotion-results flex flex-col items-center gap-2">
                {emotionPredictions
                  .filter(({ probability }) => probability > 0.0001)
                  .sort((a, b) => b.probability - a.probability)
                  .map(({ emotion, probability }) => (
                    <p key={emotion} className="text-lg leading-7 [&:not(:first-child)]:mt-6">
                      {emotionEmojis[emotion]} {emotion}: {(probability).toFixed(2)}%
                    </p>
                  ))}
              </div>
            )}
          </div>
        )}
      </div>
  );
}

export default Emote; 