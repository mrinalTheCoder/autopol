package com.autopol.deepmodel;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.tensorflow.Operation;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

public class YoloPostProcessor {
    private final static float OVERLAP_THRESHOLD = 0.5f;
    private final static double anchors[] = {1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52};
    private final static int SIZE = 13;
    private final static int MAX_RECOGNIZED_CLASSES = 13;
    private final static float THRESHOLD = 0.3f;
    private final static int MAX_RESULTS = 15;
    private final static int NUMBER_OF_BOUNDING_BOX = 5;
    private static YoloPostProcessor postProcessor;

    private YoloPostProcessor() {}

    public static YoloPostProcessor getInstance() {
        if (postProcessor == null) {
            postProcessor = new YoloPostProcessor();
        }

        return postProcessor;
    }

    public int getOutputSizeByShape(final Operation operation) {
        return (int) (operation.output(0).shape().size(3) * Math.pow(SIZE,2));
    }

    public List<DetectionResult> detectObjectsAndBoundingBox(final float[] tensorFlowOutput, final Vector<String> labels) {
        int numClass = (int) (tensorFlowOutput.length / (Math.pow(SIZE,2) * NUMBER_OF_BOUNDING_BOX) - 5);
        BoundingBox[][][] boundingBoxPerCell = new BoundingBox[SIZE][SIZE][NUMBER_OF_BOUNDING_BOX];
        PriorityQueue<DetectionResult> priorityQueue = new PriorityQueue<>(MAX_RECOGNIZED_CLASSES);

        int offset = 0;
        for (int cy=0; cy<SIZE; cy++) {        // SIZE * SIZE cells
            for (int cx=0; cx<SIZE; cx++) {
                for (int b=0; b<NUMBER_OF_BOUNDING_BOX; b++) {   // 5 bounding boxes per each cell
                    boundingBoxPerCell[cx][cy][b] = getModel(tensorFlowOutput, cx, cy, b, numClass, offset);
                    calculateTopPredictions(boundingBoxPerCell[cx][cy][b], priorityQueue, labels);
                    offset = offset + numClass + 5;
                }
            }
        }

        return getObjectsAndBoundingBoxes(priorityQueue);
    }

    private BoundingBox getModel(final float[] tensorFlowOutput, int cx, int cy, int b, int numClass, int offset) {
        BoundingBox model = new BoundingBox();
        Sigmoid sigmoid = new Sigmoid();
        model.setX((cx + sigmoid.value(tensorFlowOutput[offset])) * 32);
        model.setY((cy + sigmoid.value(tensorFlowOutput[offset + 1])) * 32);
        model.setWidth(Math.exp(tensorFlowOutput[offset + 2]) * anchors[2 * b] * 32);
        model.setHeight(Math.exp(tensorFlowOutput[offset + 3]) * anchors[2 * b + 1] * 32);
        model.setConfidence(sigmoid.value(tensorFlowOutput[offset + 4]));

        model.setClasses(new double[numClass]);

        for (int probIndex=0; probIndex<numClass; probIndex++) {
            model.getClasses()[probIndex] = tensorFlowOutput[probIndex + offset + 5];
        }

        return model;
    }

    private void calculateTopPredictions(final BoundingBox boundingBox, final PriorityQueue<DetectionResult> predictionQueue,
                                         final Vector<String> labels) {
        for (int i=0; i<boundingBox.getClasses().length; i++) {
            ArgMax.Result argMax = new ArgMax(new SoftMax(boundingBox.getClasses()).getValue()).getResult();
            double confidenceInClass = argMax.getMaxValue() * boundingBox.getConfidence();

            if (confidenceInClass > THRESHOLD) {
                predictionQueue.add(new DetectionResult(argMax.getIndex(), labels.get(argMax.getIndex()), (float) confidenceInClass,
                        new BoxPosition((float) (boundingBox.getX() - boundingBox.getWidth() / 2),
                                (float) (boundingBox.getY() - boundingBox.getHeight() / 2),
                                (float) boundingBox.getWidth(),
                                (float) boundingBox.getHeight())));
            }
        }
    }

    private List<DetectionResult> getObjectsAndBoundingBoxes(final PriorityQueue<DetectionResult> priorityQueue) {
        List<DetectionResult> detectionResults = new ArrayList();

        if (priorityQueue.size() > 0) {
            // Best recognition
            DetectionResult bestRecognition = priorityQueue.poll();
            detectionResults.add(bestRecognition);

            for (int i = 0; i < Math.min(priorityQueue.size(), MAX_RESULTS); ++i) {
                DetectionResult detectionResult = priorityQueue.poll();
                boolean overlaps = false;
                for (DetectionResult previousRecognition : detectionResults) {
                    overlaps = overlaps || (getIntersectionProportion(previousRecognition.getLocation(),
                            detectionResult.getLocation()) > OVERLAP_THRESHOLD);
                }

                if (!overlaps) {
                    detectionResults.add(detectionResult);
                }
            }
        }
        return detectionResults;
    }

    private float getIntersectionProportion(BoxPosition primaryShape, BoxPosition secondaryShape) {
        if (overlaps(primaryShape, secondaryShape)) {
            float intersectionSurface = Math.max(0, Math.min(primaryShape.getRight(), secondaryShape.getRight()) - Math.max(primaryShape.getLeft(), secondaryShape.getLeft())) *
                    Math.max(0, Math.min(primaryShape.getBottom(), secondaryShape.getBottom()) - Math.max(primaryShape.getTop(), secondaryShape.getTop()));
            float surfacePrimary = Math.abs(primaryShape.getRight() - primaryShape.getLeft()) * Math.abs(primaryShape.getBottom() - primaryShape.getTop());
            return intersectionSurface / surfacePrimary;
        }

        return 0f;

    }

    private boolean overlaps(BoxPosition primary, BoxPosition secondary) {
        return primary.getLeft() < secondary.getRight() && primary.getRight() > secondary.getLeft()
                && primary.getTop() < secondary.getBottom() && primary.getBottom() > secondary.getTop();
    }
}