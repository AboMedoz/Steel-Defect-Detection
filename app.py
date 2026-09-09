import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import cv2

from severity import get_severity, resize_mask, severity_rank


st.set_page_config(
    page_title="Surface Defect Detection",
    page_icon="🔍",
)


st.title("🔍 Surface Defect Detection")
st.write(
    "Upload a steel or wood surface image to detect defects and estimate "
    "their severity based on the affected surface area."
)


@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


st.sidebar.header("⚙️ Model Settings")

material = st.sidebar.selectbox(
    "Material",
    ["Steel", "Wood"]
)

imgsz = 640
if material == "Steel":
    model_path = "models/metal.pt"
else:
    model_path = "models/wood.pt"

try:
    model = load_model(model_path)
except Exception as e:
    st.error("❌ Failed to load the YOLO model.")
    st.code(str(e))
    st.stop()


confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05
)


st.sidebar.divider()

st.sidebar.subheader("📊 Severity Rules")
st.sidebar.write(
    "Severity is estimated using the percentage of the "
    f"{material.lower()} surface affected by the detected defect."
)

st.sidebar.write("🟢 **LOW:** < 1%")
st.sidebar.write("🟡 **MEDIUM:** 1% – < 5%")
st.sidebar.write("🟠 **HIGH:** 5% – < 10%")
st.sidebar.write("🔴 **CRITICAL:** ≥ 10%")

st.sidebar.divider()

st.sidebar.info(
    "These thresholds are project-defined heuristics and "
    "are not official industrial acceptance limits."
)


uploaded_file = st.file_uploader(
    f"📤 Upload a {material.lower()} surface image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    if st.button("🚀 Run Detection & Severity Analysis", type="primary"):
        with st.spinner("Running YOLO segmentation and severity analysis..."):
            results = model.predict(
                source=image,
                conf=confidence,
                imgsz=imgsz,
                verbose=False
            )

            result = results[0]

            annotated_image = result.plot()

            image_width, image_height = image.size

            total_pixels = image_width * image_height

            detections = []
            all_masks = []

            severity_colors = {
                "LOW": (0, 255, 0),
                "MEDIUM": (0, 255, 255),
                "HIGH": (0, 165, 255),
                "CRITICAL": (0, 0, 255)
            }

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                masks = None

                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()

                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i])

                    confidence_score = float(boxes.conf[i])

                    class_name = model.names[class_id]

                    if masks is not None and i < len(masks):
                        binary_mask = resize_mask(
                            masks[i],
                            image_width,
                            image_height
                        )

                        defect_pixels = int(
                            np.count_nonzero(binary_mask)
                        )

                        area_percent = (
                            defect_pixels / total_pixels
                        ) * 100

                        all_masks.append(binary_mask)

                    else:
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()

                        box_area = (
                            max(0, x2 - x1) *
                            max(0, y2 - y1)
                        )

                        area_percent = (
                            box_area / total_pixels
                        ) * 100

                    severity, symbol = get_severity(area_percent)

                    detections.append(
                        {
                            "Defect": class_name,
                            "Confidence": confidence_score,
                            "Area (%)": area_percent,
                            "Severity": severity,
                            "Symbol": symbol
                        }
                    )

                    # Highlight the defect according to its severity
                    color = severity_colors[severity]

                    if masks is not None and i < len(masks):
                        overlay = annotated_image.copy()

                        overlay[binary_mask > 0] = color

                        annotated_image = cv2.addWeighted(
                            annotated_image,
                            0.7,
                            overlay,
                            0.3,
                            0
                        )

                    # Add severity text to the detection
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()

                    x1 = int(x1)
                    y1 = int(y1)

                    label = f"{severity} | {area_percent:.2f}%"

                    cv2.putText(
                        annotated_image,
                        label,
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                        cv2.LINE_AA
                    )

            # Convert BGR -> RGB for Streamlit
            annotated_image = annotated_image[:, :, ::-1]

            if len(all_masks) > 0:
                combined_mask = np.any(
                    np.stack(all_masks),
                    axis=0
                )

                total_defect_pixels = int(
                    np.count_nonzero(combined_mask)
                )

                total_coverage = (
                    total_defect_pixels / total_pixels
                ) * 100

            elif len(detections) > 0:
                total_coverage = min(
                    sum(d["Area (%)"] for d in detections),
                    100
                )

            else:
                total_coverage = 0.0

            if len(detections) > 0:
                area_severity, area_symbol = get_severity(
                    total_coverage
                )

                worst_individual = max(
                    detections,
                    key=lambda x: severity_rank(x["Severity"])
                )

                worst_severity = worst_individual["Severity"]

                if severity_rank(worst_severity) > severity_rank(
                    area_severity
                ):
                    overall_severity = worst_severity

                    overall_symbol = {
                        "LOW": "🟢",
                        "MEDIUM": "🟡",
                        "HIGH": "🟠",
                        "CRITICAL": "🔴"
                    }[overall_severity]

                else:
                    overall_severity = area_severity
                    overall_symbol = area_symbol

            else:
                overall_severity = "NONE"
                overall_symbol = "⚪"

        st.subheader("🖼️ Detection Results")

        input_col, output_col = st.columns(2)

        with input_col:
            st.markdown("### 📷 Input Image")
            st.image(image, use_container_width=True)

        with output_col:
            st.markdown("### 🎯 Detection Result")
            st.image(annotated_image, use_container_width=True)

        st.divider()

        st.subheader("📊 Overall Severity")

        if len(detections) > 0:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Defects Detected",
                    len(detections)
                )

            with col2:
                st.metric(
                    "Defect Coverage",
                    f"{total_coverage:.2f}%"
                )

            with col3:
                st.metric(
                    "Overall Severity",
                    f"{overall_symbol} {overall_severity}"
                )

            with col4:
                average_confidence = np.mean(
                    [d["Confidence"] for d in detections]
                )

                st.metric(
                    "Average Confidence",
                    f"{average_confidence:.2f}"
                )

            if overall_severity == "LOW":
                st.success("🟢 Overall Severity: **LOW**")

            elif overall_severity == "MEDIUM":
                st.warning("🟡 Overall Severity: **MEDIUM**")

            elif overall_severity == "HIGH":
                st.warning("🟠 Overall Severity: **HIGH**")

            elif overall_severity == "CRITICAL":
                st.error("🔴 Overall Severity: **CRITICAL**")

        else:
            st.success("⚪ No defects were detected.")

        if len(detections) > 0:
            st.divider()
            st.subheader("📋 Defect Summary")

            df = pd.DataFrame(detections)

            grouped = (
                df.groupby("Defect")
                .agg(
                    Occurrences=("Defect", "count"),
                    Total_Area=("Area (%)", "sum"),
                    Average_Confidence=(
                        "Confidence",
                        "mean"
                    )
                )
                .reset_index()
            )

            grouped["Severity"] = grouped["Total_Area"].apply(
                lambda x: get_severity(x)[0]
            )

            grouped["Symbol"] = grouped["Severity"].map(
                {
                    "LOW": "🟢",
                    "MEDIUM": "🟡",
                    "HIGH": "🟠",
                    "CRITICAL": "🔴"
                }
            )

            display_df = grouped[
                [
                    "Symbol",
                    "Defect",
                    "Occurrences",
                    "Total_Area",
                    "Average_Confidence",
                    "Severity"
                ]
            ].copy()

            display_df.columns = [
                "",
                "Defect",
                "Occurrences",
                "Total Area (%)",
                "Average Confidence",
                "Severity"
            ]

            display_df["Total Area (%)"] = display_df[
                "Total Area (%)"
            ].round(2)

            display_df["Average Confidence"] = display_df[
                "Average Confidence"
            ].round(2)

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

        if len(detections) > 0:
            st.divider()
            st.subheader("🔎 Individual Defects")

            for index, detection in enumerate(
                detections,
                start=1
            ):
                st.markdown(
                    f"### {index}. "
                    f"{detection['Symbol']} "
                    f"{detection['Defect']}"
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.write(
                        f"**Confidence:** "
                        f"{detection['Confidence']:.2f}"
                    )

                with c2:
                    st.write(
                        f"**Affected Area:** "
                        f"{detection['Area (%)']:.2f}%"
                    )

                with c3:
                    st.write(
                        f"**Severity:** "
                        f"{detection['Severity']}"
                    )

                st.divider()

        if len(detections) > 0:
            st.subheader("📥 Inspection Report")

            report_df = pd.DataFrame(
                [
                    {
                        "Defect Number": i,
                        "Defect": d["Defect"],
                        "Confidence": round(
                            d["Confidence"],
                            4
                        ),
                        "Area (%)": round(
                            d["Area (%)"],
                            4
                        ),
                        "Severity": d["Severity"]
                    }

                    for i, d in enumerate(
                        detections,
                        start=1
                    )
                ]
            )

            csv_data = report_df.to_csv(index=False)

            st.download_button(
                label="📥 Download Defect Report (CSV)",
                data=csv_data,
                file_name=f"{material.lower()}_defect_report.csv",
                mime="text/csv"
            )

else:
    st.info(
        f"👆 Upload a {material.lower()} surface image above "
        "to start the analysis.")