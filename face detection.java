package com.example.facemaskdetection;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

public class MaskDetector {

    private static final String MODEL_PATH = "face_mask_detector.tflite";
    private static final String[] LABELS = {"No Mask", "Mask"}; // Adjust based on your model's output
    private static final int INPUT_SIZE = 160; // Or whatever your model expects (e.g., 224x224, 300x300)
    private static final float NORM_MEAN = 127.5f;
    private static final float NORM_STD = 127.5f;
    private static final float CONFIDENCE_THRESHOLD = 0.6f; // Minimum confidence to show a detection

    private Interpreter tflite;
    private ImageProcessor imageProcessor;
    private TensorProcessor probabilityProcessor;

    public static class DetectionResult {
        public RectF location;
        public String label;
        public float confidence;
    }

    public MaskDetector(Context context) throws IOException {
        // Load the TFLite model
        tflite = new Interpreter(FileUtil.loadMappedFile(context, MODEL_PATH));

        // Initialize ImageProcessor for preprocessing input images
        imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(NORM_MEAN, NORM_STD)) // Normalize to [-1, 1] if your model expects it
                .build();

        // Initialize TensorProcessor for post-processing model output (if needed for probabilities)
        probabilityProcessor = new TensorProcessor.Builder()
                .add(new NormalizeOp(0.0f, 255.0f)) // Example, adjust based on your model's output
                .build();
    }

    public List<DetectionResult> detect(Bitmap bitmap) {
        if (tflite == null) {
            Log.e("MaskDetector", "Interpreter not initialized.");
            return new ArrayList<>();
        }

        // Create a TensorImage from the Bitmap
        TensorImage tensorImage = new TensorImage(DataType.UINT8); // Or FLOAT32 depending on model input
        tensorImage.load(bitmap);

        // Preprocess the image
        tensorImage = imageProcessor.process(tensorImage);

        // Prepare output buffers
        // Assuming your model outputs:
        // Output 0: float array of detections (num_detections, 4) - bounding box [ymin, xmin, ymax, xmax]
        // Output 1: float array of class labels (num_detections) - class ID (0 for no mask, 1 for mask)
        // Output 2: float array of scores (num_detections) - confidence scores
        // Output 3: float num_detections (scalar) - actual number of valid detections

        // Adjust these buffer sizes and types based on your specific TFLite model's outputs
        int numDetections = 10; // Max number of detections your model outputs
        TensorBuffer outputLocations = TensorBuffer.createFixedSize(new int[]{1, numDetections, 4}, DataType.FLOAT32);
        TensorBuffer outputClasses = TensorBuffer.createFixedSize(new int[]{1, numDetections}, DataType.FLOAT32);
        TensorBuffer outputScores = TensorBuffer.createFixedSize(new int[]{1, numDetections}, DataType.FLOAT32);
        TensorBuffer outputNumDetections = TensorBuffer.createFixedSize(new int[]{1}, DataType.FLOAT32);

        // Run inference
        tflite.runForMultipleInputsOutputs(
                new Object[]{tensorImage.getBuffer()},
                new java.util.HashMap<Integer, Object>() {{
                    put(0, outputLocations.getBuffer());
                    put(1, outputClasses.getBuffer());
                    put(2, outputScores.getBuffer());
                    put(3, outputNumDetections.getBuffer());
                }}
        );

        List<DetectionResult> results = new ArrayList<>();
        float[] locations = outputLocations.getFloatArray();
        float[] classes = outputClasses.getFloatArray();
        float[] scores = outputScores.getFloatArray();
        int actualNumDetections = (int) outputNumDetections.getFloatArray()[0];

        // Process results
        for (int i = 0; i < actualNumDetections; ++i) {
            float score = scores[i];
            if (score > CONFIDENCE_THRESHOLD) {
                DetectionResult result = new DetectionResult();
                result.confidence = score;
                result.label = LABELS[(int) classes[i]]; // Map class ID to label

                // Bounding box coordinates are normalized [0, 1] and in [ymin, xmin, ymax, xmax] format
                float ymin = locations[i * 4] * bitmap.getHeight();
                float xmin = locations[i * 4 + 1] * bitmap.getWidth();
                float ymax = locations[i * 4 + 2] * bitmap.getHeight();
                float xmax = locations[i * 4 + 3] * bitmap.getWidth();

                result.location = new RectF(xmin, ymin, xmax, ymax);
                results.add(result);
            }
        }
        return results;
    }

    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
    }
}