package com.autopol;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.media.Image;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;

import com.autopol.deepmodel.DetectionResult;
import com.autopol.deepmodel.YoloObjectDetector;
import com.autopol.utils.ImageUtils;
import com.autopol.customview.OverlayView;

import java.util.List;

public class ObstacleDetectorActivity extends CameraActivity implements OnImageAvailableListener {
    private static int MODEL_IMAGE_INPUT_SIZE = 416;
    private static String LOGGING_TAG = "autopol";
    private static float TEXT_SIZE_DIP = 10;

    private Integer sensorOrientation;
    private int previewWidth = 0;
    private int previewHeight = 0;

    private YoloObjectDetector yoloObjectDetector;
    private Bitmap imageBitmapForModel = null;
    private boolean computing = false;
    private Matrix imageTransformMatrix;

    private OverlayView overlayView;

    @Override
    public void onPreviewSizeChosen(final Size previewSize, final int rotation) {
        final float textSizePx = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP,
                TEXT_SIZE_DIP, getResources().getDisplayMetrics());

        yoloObjectDetector = YoloObjectDetector.create(getAssets());
        overlayView = (OverlayView) findViewById(R.id.overlay);

        final int screenOrientation = getWindowManager().getDefaultDisplay().getRotation();
        //Sensor orientation: 90, Screen orientation: 0
        sensorOrientation = rotation + screenOrientation;
        Log.i(LOGGING_TAG, String.format("Camera rotation: %d, Screen orientation: %d, Sensor orientation: %d",
                rotation, screenOrientation, sensorOrientation));

        previewWidth = previewSize.getWidth();
        previewHeight = previewSize.getHeight();
        Log.i(LOGGING_TAG, String.format("Initializing at size %dx%d", previewWidth, previewHeight));
        // create empty bitmap
        imageBitmapForModel = Bitmap.createBitmap(MODEL_IMAGE_INPUT_SIZE, MODEL_IMAGE_INPUT_SIZE, Config.ARGB_8888);
        imageTransformMatrix = ImageUtils.getTransformationMatrix(previewWidth, previewHeight,
                MODEL_IMAGE_INPUT_SIZE, MODEL_IMAGE_INPUT_SIZE, sensorOrientation,true);
        imageTransformMatrix.invert(new Matrix());
    }

    @Override
    public void onImageAvailable(final ImageReader reader) {
        Image imageFromCamera = null;

        try {
            imageFromCamera = reader.acquireLatestImage();
            if (imageFromCamera == null) {
                return;
            }
            if (computing) {
                imageFromCamera.close();
                return;
            }
            computing = true;
            preprocessImageForModel(imageFromCamera);
            imageFromCamera.close();
        } catch (final Exception ex) {
            if (imageFromCamera != null) {
                imageFromCamera.close();
            }
            Log.e(LOGGING_TAG, ex.getMessage());
        }

        runInBackground(() -> {
            final List<DetectionResult> results = yoloObjectDetector.detectObjects(imageBitmapForModel);
            overlayView.setResults(results);
            /*
            if(results.size() > 0) {
                String title = results.get(0).getTitle();
                for(int ix = 1 ; ix < results.size() - 1; ix++) {
                    title += ", ";
                    title += results.get(ix).getTitle();
                }
                Toast.makeText(ObstacleDetectorActivity.this, title, Toast.LENGTH_LONG).show();
            }
            */
            requestRender();
            computing = false;
        });
    }

    private void preprocessImageForModel(final Image imageFromCamera) {
        Bitmap rgbBitmapForCameraImage = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        rgbBitmapForCameraImage.setPixels(ImageUtils.convertYUVToARGB(imageFromCamera, previewWidth, previewHeight),
                0, previewWidth, 0, 0, previewWidth, previewHeight);
        new Canvas(imageBitmapForModel).drawBitmap(rgbBitmapForCameraImage, imageTransformMatrix, null);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (yoloObjectDetector != null) {
            yoloObjectDetector.close();
        }
    }
}
