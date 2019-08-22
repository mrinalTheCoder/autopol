package com.autopol.deepmodel;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Vector;

public class YoloObjectDetector {
    private static int MODEL_IMAGE_INPUT_SIZE = 416;
    private static int IMAGE_MEAN = 128;
    private static float IMAGE_STD = 128.0f;
    private static String INPUT_NODE_NAME = "input";
    private static String OUTPUT_NODE_NAME = "output";
    private static String MODEL_FILE_NAME = "tiny_yolo.pb";
    private static String MODEL_FILE_PATH = "file:///android_asset/" + MODEL_FILE_NAME;
    private String LABEL_FILE_NAME = "tiny_yolo_labels.txt";

    private int outputSize;
    private Vector<String> labels = new Vector();
    private TensorFlowInferenceInterface inferenceInterface;

    private YoloObjectDetector(final AssetManager assetManager) {
        init(assetManager);
    }

    private void init(final AssetManager assetManager) {
        inferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE_PATH);
        outputSize = YoloPostProcessor.getInstance()
                .getOutputSizeByShape(inferenceInterface.graphOperation(OUTPUT_NODE_NAME));

        try (BufferedReader br = new BufferedReader(new InputStreamReader(assetManager.open(LABEL_FILE_NAME)))) {
            String line;
            while ((line = br.readLine()) != null) {
                labels.add(line);
            }
        } catch (IOException ex) {
            throw new RuntimeException("Problem reading label file!", ex);
        }
    }

    public static YoloObjectDetector create(final AssetManager assetManager) {
        return new YoloObjectDetector(assetManager);
    }

    public void close() {
        inferenceInterface.close();
    }

    public List<DetectionResult> detectObjects(final Bitmap inputImage) {
        return YoloPostProcessor.getInstance().detectObjectsAndBoundingBox(runTensorFlowWithYoloModel(inputImage), labels);
    }

    private float[] runTensorFlowWithYoloModel(final Bitmap inputImage) {
        final float[] tfOutput = new float[outputSize];
        inferenceInterface.feed(INPUT_NODE_NAME, preprocess(inputImage), 1, MODEL_IMAGE_INPUT_SIZE, MODEL_IMAGE_INPUT_SIZE, 3);
        inferenceInterface.run(new String[]{OUTPUT_NODE_NAME});
        inferenceInterface.fetch(OUTPUT_NODE_NAME, tfOutput);

        return tfOutput;
    }

    private float[] preprocess(final Bitmap inputImage) {
        int[] intValues = new int[MODEL_IMAGE_INPUT_SIZE * MODEL_IMAGE_INPUT_SIZE];
        float[] floatValues = new float[MODEL_IMAGE_INPUT_SIZE * MODEL_IMAGE_INPUT_SIZE * 3];

        inputImage.getPixels(intValues, 0, inputImage.getWidth(), 0, 0, inputImage.getWidth(), inputImage.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
            floatValues[i * 3 + 2] = ((val & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
        }
        return floatValues;
    }
}
