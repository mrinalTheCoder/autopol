package com.autopol.deepmodel;

public final class DetectionResult implements Comparable<DetectionResult> {
    private final Integer id;
    private final String title;
    private final Float confidence;
    private BoxPosition location;

    public DetectionResult(final Integer id, final String title,
                           final Float confidence, final BoxPosition location) {
        this.id = id;
        this.title = title;
        this.confidence = confidence;
        this.location = location;
    }

    public Integer getId() {
        return id;
    }

    public String getTitle() {
        return title;
    }

    public Float getConfidence() {
        return confidence;
    }

    public BoxPosition getLocation() {
        return new BoxPosition(location);
    }

    public void setLocation(BoxPosition location) {
        this.location = location;
    }

    @Override
    public int compareTo (DetectionResult other) {
        return Float.compare(other.getConfidence(), this.getConfidence());
    }

    @Override
    public String toString() {
        return "DetectionResult{" +
                "id=" + id +
                ", title='" + title + '\'' +
                ", confidence=" + confidence +
                ", location=" + location +
                '}';
    }
}
