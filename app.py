import streamlit as st
import rasterio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import folium
from streamlit_folium import st_folium
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import io
from PIL import Image
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from scipy import stats

st.set_page_config(
    page_title="Surface Water Extent Mapping",
    page_icon="🌊",
    layout="wide"
)

@st.cache_data
def load_tif_files():
    """Load all TIF files from attached_assets directory"""
    tif_files = list(Path("attached_assets").glob("*.tif"))
    
    data_dict = {}
    
    for tif_file in sorted(tif_files):
        try:
            with rasterio.open(tif_file) as src:
                img_data = src.read()
                bounds = src.bounds
                crs = src.crs
                transform = src.transform
                
                filename = tif_file.stem
                parts = filename.split('_')
                if len(parts) >= 2:
                    month_str = parts[1]
                    
                    data_dict[month_str] = {
                        'data': img_data,
                        'bounds': bounds,
                        'crs': crs,
                        'transform': transform,
                        'path': str(tif_file),
                        'filename': filename
                    }
        except Exception as e:
            st.warning(f"Could not load {tif_file.name}: {str(e)}")
    
    return data_dict

def extract_water_extent(img_data, method='NDWI'):
    """
    Extract water extent from satellite imagery using spectral indices
    
    Parameters:
    - img_data: Raster data array
    - method: 'NDWI', 'MNDWI', or 'Threshold'
    
    Returns:
    - water_mask: Boolean array indicating water pixels
    - water_pixels: Count of water pixels
    - water_percentage: Percentage of water coverage
    """
    if img_data.shape[0] >= 3:
        green = img_data[1].astype(float)
        
        if method == 'MNDWI' and img_data.shape[0] >= 6:
            swir = img_data[5].astype(float)
            index = np.zeros_like(green)
            denominator = green + swir
            mask = denominator != 0
            index[mask] = (green[mask] - swir[mask]) / denominator[mask]
            water_mask = index > 0.1
        elif method == 'NDWI':
            nir = img_data[3].astype(float) if img_data.shape[0] > 3 else img_data[0].astype(float)
            index = np.zeros_like(green)
            denominator = green + nir
            mask = denominator != 0
            index[mask] = (green[mask] - nir[mask]) / denominator[mask]
            water_mask = index > 0.0
        else:
            band = img_data[0].astype(float)
            threshold = np.percentile(band[band > 0], 30)
            water_mask = band < threshold
    else:
        band = img_data[0].astype(float)
        threshold = np.percentile(band[band > 0], 30)
        water_mask = band < threshold
    
    water_pixels = np.sum(water_mask)
    total_pixels = water_mask.size
    water_percentage = (water_pixels / total_pixels) * 100
    
    return water_mask, water_pixels, water_percentage

def calculate_flood_risk(water_percentages):
    """Calculate flood risk based on water extent trends"""
    if len(water_percentages) < 2:
        return "Insufficient Data", "gray"
    
    mean_extent = np.mean(water_percentages)
    std_extent = np.std(water_percentages)
    current_extent = water_percentages[-1]
    
    if current_extent > mean_extent + 1.5 * std_extent:
        return "High Risk", "red"
    elif current_extent > mean_extent + 0.5 * std_extent:
        return "Moderate Risk", "orange"
    elif current_extent > mean_extent:
        return "Low Risk", "yellow"
    else:
        return "Normal", "green"

def create_water_visualization(img_data, water_mask):
    """Create a visualization combining satellite imagery and water detection"""
    if img_data.shape[0] >= 3:
        rgb_bands = []
        for i in range(3):
            band = img_data[i].astype(float)
            band_max = np.nanmax(band) if np.any(band > 0) else 1.0
            normalized = np.nan_to_num(band / band_max * 255, nan=0, posinf=255, neginf=0)
            rgb_bands.append(np.clip(normalized, 0, 255).astype(np.uint8))
        rgb = np.dstack(rgb_bands)
    else:
        band = img_data[0].astype(float)
        band_max = np.nanmax(band) if np.any(band > 0) else 1.0
        normalized = np.nan_to_num(band / band_max * 255, nan=0, posinf=255, neginf=0)
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        rgb = np.dstack([normalized, normalized, normalized])
    
    overlay = rgb.copy()
    overlay[water_mask] = [0, 100, 255]
    
    blended = (0.6 * rgb + 0.4 * overlay).astype(np.uint8)
    
    return blended

def generate_pdf_report(months, percentages, water_data, data_dict, classification_method):
    """Generate a comprehensive PDF report with water extent analysis and flood risk assessment"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    story.append(Paragraph("Surface Water Extent Mapping Report", title_style))
    story.append(Paragraph("Chembarambakkam Lake - Flood Risk Analysis", styles['Heading3']))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"Analysis Period: {months[0]} to {months[-1]}", styles['Normal']))
    story.append(Paragraph(f"Classification Method: {classification_method}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Executive Summary", heading_style))
    summary_data = [
        ['Metric', 'Value'],
        ['Total Months Analyzed', str(len(months))],
        ['Average Water Extent', f"{np.mean(percentages):.2f}%"],
        ['Maximum Extent', f"{np.max(percentages):.2f}% ({months[np.argmax(percentages)]})"],
        ['Minimum Extent', f"{np.min(percentages):.2f}% ({months[np.argmin(percentages)]})"],
        ['Standard Deviation', f"{np.std(percentages):.2f}%"],
        ['Volatility Index', f"{(np.std(percentages) / np.mean(percentages) * 100):.1f}%"]
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))
    
    risk_level, risk_color = calculate_flood_risk(percentages)
    story.append(Paragraph("Current Flood Risk Assessment", heading_style))
    story.append(Paragraph(f"<b>Risk Level:</b> {risk_level}", styles['Normal']))
    story.append(Paragraph(f"<b>Latest Month:</b> {months[-1]} - {percentages[-1]:.2f}%", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    if risk_level == "High Risk":
        story.append(Paragraph("⚠️ HIGH FLOOD RISK DETECTED! Immediate monitoring and preparedness measures recommended.", styles['Normal']))
    elif risk_level == "Moderate Risk":
        story.append(Paragraph("⚡ Moderate flood risk. Continue monitoring the situation closely.", styles['Normal']))
    else:
        story.append(Paragraph("✓ Water levels within normal or low risk range.", styles['Normal']))
    
    story.append(PageBreak())
    
    story.append(Paragraph("Detailed Monthly Water Extent Data", heading_style))
    monthly_data = [['Month', 'Water Extent (%)', 'Water Pixels', 'Change from Previous']]
    for i, month in enumerate(months):
        change = percentages[i] - percentages[i-1] if i > 0 else 0
        monthly_data.append([
            month,
            f"{percentages[i]:.2f}",
            f"{water_data[month]['pixels']:,}",
            f"{change:+.2f}%" if i > 0 else "—"
        ])
    
    monthly_table = Table(monthly_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    monthly_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9)
    ]))
    story.append(monthly_table)
    story.append(Spacer(1, 0.3*inch))
    
    if len(months) >= 3:
        story.append(PageBreak())
        story.append(Paragraph("Predictive Flood Modeling", heading_style))
        
        percentages_array = np.array(percentages)
        months_numeric = np.arange(len(months))
        slope, intercept, r_value, p_value, std_err = stats.linregress(months_numeric, percentages_array)
        
        future_months_numeric = np.arange(len(months), len(months) + 3)
        future_predictions = slope * future_months_numeric + intercept
        
        forecast_data = [['Period', 'Predicted Extent (%)', 'Risk Level']]
        mean_val = np.mean(percentages)
        std_val = np.std(percentages)
        
        for i, pred in enumerate(future_predictions):
            risk = 'High' if pred > mean_val + 1.5 * std_val else 'Moderate' if pred > mean_val + 0.5 * std_val else 'Normal'
            forecast_data.append([f"Month +{i+1}", f"{pred:.2f}", risk])
        
        forecast_table = Table(forecast_data, colWidths=[2*inch, 2*inch, 2*inch])
        forecast_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff7f0e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(forecast_table)
        story.append(Spacer(1, 0.2*inch))
        
        trend_direction = "Increasing" if slope > 0 else "Decreasing"
        story.append(Paragraph(f"<b>Trend Direction:</b> {trend_direction} ({abs(slope):.2f}% per month)", styles['Normal']))
        story.append(Paragraph(f"<b>Model Confidence (R²):</b> {r_value**2:.3f}", styles['Normal']))
    
    story.append(PageBreak())
    story.append(Paragraph("Recommendations", heading_style))
    
    if risk_level == "High Risk" or (len(months) >= 3 and np.max(future_predictions) > np.mean(percentages) + 1.5 * np.std(percentages)):
        story.append(Paragraph("• Implement immediate flood preparedness measures", styles['Normal']))
        story.append(Paragraph("• Increase monitoring frequency to daily", styles['Normal']))
        story.append(Paragraph("• Alert downstream communities and stakeholders", styles['Normal']))
        story.append(Paragraph("• Review and activate emergency response plans", styles['Normal']))
    elif risk_level == "Moderate Risk":
        story.append(Paragraph("• Enhance monitoring protocols", styles['Normal']))
        story.append(Paragraph("• Prepare flood response resources", styles['Normal']))
        story.append(Paragraph("• Communicate status to relevant authorities", styles['Normal']))
    else:
        story.append(Paragraph("• Continue routine monitoring", styles['Normal']))
        story.append(Paragraph("• Maintain standard operating procedures", styles['Normal']))
        story.append(Paragraph("• Regular data review and analysis", styles['Normal']))
    
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Report End", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def create_interactive_map(data_dict, water_data, selected_months):
    """Create an interactive Folium map with water extent overlays"""
    if not selected_months:
        return None
    
    first_month = selected_months[0]
    bounds = data_dict[first_month]['bounds']
    
    center_lat = (bounds.bottom + bounds.top) / 2
    center_lon = (bounds.left + bounds.right) / 2
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'darkred', 'lightred', 'beige', 
              'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 
              'lightgreen', 'gray', 'black', 'lightgray']
    
    for idx, month in enumerate(selected_months):
        if month not in data_dict:
            continue
            
        bounds = data_dict[month]['bounds']
        water_mask = water_data[month]['mask']
        water_percentage = water_data[month]['percentage']
        
        water_img = np.zeros((*water_mask.shape, 4), dtype=np.uint8)
        water_img[water_mask] = [0, 100, 255, 180]
        
        img_pil = Image.fromarray(water_img, 'RGBA')
        
        img_buffer = io.BytesIO()
        img_pil.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        
        overlay_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
        
        color = colors[idx % len(colors)]
        
        folium.raster_layers.ImageOverlay(
            image=f'data:image/png;base64,{img_base64}',
            bounds=overlay_bounds,
            opacity=0.6,
            name=f'{month} - {water_percentage:.2f}%',
            show=idx == 0
        ).add_to(m)
        
        folium.Rectangle(
            bounds=overlay_bounds,
            color=color,
            fill=False,
            weight=2,
            popup=f"{month}<br>Water Extent: {water_percentage:.2f}%<br>Pixels: {water_data[month]['pixels']:,}"
        ).add_to(m)
    
    folium.LayerControl(collapsed=False).add_to(m)
    
    return m

def main():
    st.title("🌊 Surface Water Extent Mapping")
    st.markdown("### Chembarambakkam Lake - Flood Risk Analysis")
    st.markdown("Analyzing satellite imagery from Google Earth Engine (Jan 2024 - Jan 2025)")
    
    st.divider()
    
    data_dict = load_tif_files()
    
    if not data_dict:
        st.error("No TIF files found in attached_assets directory. Please upload satellite imagery files.")
        st.info("Expected file format: Chembarambakkam_YYYY-MM_*.tif")
        return
    
    months = sorted(data_dict.keys())
    st.success(f"✅ Loaded {len(months)} months of satellite imagery")
    
    with st.sidebar:
        st.header("⚙️ Analysis Settings")
        
        classification_method = st.selectbox(
            "Water Classification Method",
            options=['NDWI', 'MNDWI', 'Threshold'],
            index=0,
            help="NDWI: Normalized Difference Water Index (uses NIR)\nMNDWI: Modified NDWI (uses SWIR, better for built-up areas)\nThreshold: Simple threshold-based classification"
        )
        
        st.info(f"Using **{classification_method}** for water detection")
        
        if classification_method == 'NDWI':
            st.markdown("**NDWI** = (Green - NIR) / (Green + NIR)")
            st.markdown("Best for general water body detection")
        elif classification_method == 'MNDWI':
            st.markdown("**MNDWI** = (Green - SWIR) / (Green + SWIR)")
            st.markdown("Better for distinguishing water from built-up areas")
        else:
            st.markdown("**Threshold-based** classification")
            st.markdown("Uses percentile-based threshold")
        
        st.divider()
        st.header("📄 Report Generation")
    
    water_data = {}
    for month, data in data_dict.items():
        water_mask, water_pixels, water_percentage = extract_water_extent(data['data'], method=classification_method)
        water_data[month] = {
            'mask': water_mask,
            'pixels': water_pixels,
            'percentage': water_percentage
        }
    
    percentages = [water_data[m]['percentage'] for m in months]
    
    with st.sidebar:
        pdf_buffer = generate_pdf_report(months, percentages, water_data, data_dict, classification_method)
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name=f"water_extent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            help="Download comprehensive analysis report with statistics, trends, and flood risk assessment"
        )
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview Dashboard",
        "🗺️ Monthly Water Extent Maps",
        "📈 Trend Analysis",
        "⚠️ Flood Risk Assessment",
        "🔮 Predictive Modeling"
    ])
    
    with tab1:
        st.subheader("Water Extent Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Months Analyzed",
                len(months),
                delta=None
            )
        
        with col2:
            st.metric(
                "Average Water Extent",
                f"{np.mean(percentages):.2f}%",
                delta=None
            )
        
        with col3:
            max_month = months[np.argmax(percentages)]
            st.metric(
                "Maximum Extent",
                f"{np.max(percentages):.2f}%",
                delta=f"{max_month}"
            )
        
        with col4:
            min_month = months[np.argmin(percentages)]
            st.metric(
                "Minimum Extent",
                f"{np.min(percentages):.2f}%",
                delta=f"{min_month}"
            )
        
        st.markdown("---")
        
        df = pd.DataFrame({
            'Month': months,
            'Water Extent (%)': percentages,
            'Water Pixels': [water_data[m]['pixels'] for m in months]
        })
        
        fig = px.bar(
            df,
            x='Month',
            y='Water Extent (%)',
            title='Water Extent by Month',
            color='Water Extent (%)',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
        
        st.dataframe(df, width='stretch')
    
    with tab2:
        st.subheader("Interactive Water Extent Visualization")
        
        selected_month = st.selectbox(
            "Select Month to Visualize",
            months,
            index=len(months)-1
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### Satellite Imagery - {selected_month}")
            img_data = data_dict[selected_month]['data']
            
            if img_data.shape[0] >= 3:
                rgb_bands_display = []
                for i in range(3):
                    band = img_data[i].astype(float)
                    band_max = np.nanmax(band) if np.any(band > 0) else 1.0
                    normalized = np.nan_to_num(band / band_max * 255, nan=0, posinf=255, neginf=0)
                    rgb_bands_display.append(np.clip(normalized, 0, 255).astype(np.uint8))
                rgb_display = np.dstack(rgb_bands_display)
            else:
                band = img_data[0].astype(float)
                band_max = np.nanmax(band) if np.any(band > 0) else 1.0
                normalized = np.nan_to_num(band / band_max * 255, nan=0, posinf=255, neginf=0)
                normalized = np.clip(normalized, 0, 255).astype(np.uint8)
                rgb_display = np.dstack([normalized, normalized, normalized])
            
            st.image(rgb_display, caption=f"Original Imagery - {selected_month}", width='stretch')
        
        with col2:
            st.markdown(f"#### Water Detection - {selected_month}")
            water_mask = water_data[selected_month]['mask']
            blended = create_water_visualization(img_data, water_mask)
            st.image(blended, caption=f"Water Extent (Blue overlay) - {selected_month}", width='stretch')
        
        st.metric(
            f"Water Extent for {selected_month}",
            f"{water_data[selected_month]['percentage']:.2f}%",
            delta=f"{water_data[selected_month]['pixels']:,} pixels"
        )
        
        st.markdown("---")
        st.markdown("#### Compare Multiple Months")
        
        compare_months = st.multiselect(
            "Select months to compare (up to 4)",
            months,
            default=months[-min(2, len(months)):]
        )
        
        if compare_months:
            cols = st.columns(min(len(compare_months), 4))
            for idx, month in enumerate(compare_months[:4]):
                with cols[idx]:
                    img_data = data_dict[month]['data']
                    water_mask = water_data[month]['mask']
                    blended = create_water_visualization(img_data, water_mask)
                    st.image(blended, caption=f"{month}\n{water_data[month]['percentage']:.2f}%", width='stretch')
        
        st.markdown("---")
        st.markdown("#### 🗺️ Interactive Geospatial Map")
        st.info("Pan, zoom, and toggle layers to explore water extent changes across different months")
        
        map_months = st.multiselect(
            "Select months to display on the map",
            months,
            default=months[-min(3, len(months)):],
            key="map_months_selector"
        )
        
        if map_months:
            interactive_map = create_interactive_map(data_dict, water_data, map_months)
            if interactive_map:
                st_folium(interactive_map, width=None, height=500)
        else:
            st.warning("Please select at least one month to display on the map")
    
    with tab3:
        st.subheader("Temporal Trend Analysis")
        
        df_trend = pd.DataFrame({
            'Month': months,
            'Water Extent (%)': percentages
        })
        
        fig_line = go.Figure()
        
        fig_line.add_trace(go.Scatter(
            x=df_trend['Month'],
            y=df_trend['Water Extent (%)'],
            mode='lines+markers',
            name='Water Extent',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10)
        ))
        
        mean_val = df_trend['Water Extent (%)'].mean()
        fig_line.add_hline(
            y=mean_val,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Average: {mean_val:.2f}%"
        )
        
        fig_line.update_layout(
            title='Water Extent Trend Over Time',
            xaxis_title='Month',
            yaxis_title='Water Extent (%)',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_line, width='stretch')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Statistical Summary")
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'],
                'Value (%)': [
                    f"{np.mean(percentages):.2f}",
                    f"{np.median(percentages):.2f}",
                    f"{np.std(percentages):.2f}",
                    f"{np.min(percentages):.2f}",
                    f"{np.max(percentages):.2f}",
                    f"{np.max(percentages) - np.min(percentages):.2f}"
                ]
            })
            st.dataframe(stats_df, width='stretch', hide_index=True)
        
        with col2:
            st.markdown("#### Month-over-Month Change")
            if len(percentages) > 1:
                changes = [0] + [percentages[i] - percentages[i-1] for i in range(1, len(percentages))]
                change_df = pd.DataFrame({
                    'Month': months,
                    'Change (%)': [f"{c:+.2f}" for c in changes],
                    'Direction': ['—' if c == 0 else '📈' if c > 0 else '📉' for c in changes]
                })
                st.dataframe(change_df, width='stretch', hide_index=True)
    
    with tab4:
        st.subheader("Flood Risk Assessment")
        
        risk_level, risk_color = calculate_flood_risk(percentages)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"### Current Risk Level")
            st.markdown(f"## :{risk_color}[{risk_level}]")
            
            st.markdown("---")
            
            if len(percentages) >= 2:
                latest_month = months[-1]
                latest_extent = percentages[-1]
                mean_extent = np.mean(percentages)
                
                st.metric(
                    f"Latest Month ({latest_month})",
                    f"{latest_extent:.2f}%",
                    delta=f"{latest_extent - mean_extent:+.2f}% vs average"
                )
                
                st.markdown("#### Risk Thresholds")
                st.markdown(f"- **Normal:** Below {mean_extent:.2f}%")
                st.markdown(f"- **Low Risk:** {mean_extent:.2f}% - {mean_extent + 0.5 * np.std(percentages):.2f}%")
                st.markdown(f"- **Moderate Risk:** {mean_extent + 0.5 * np.std(percentages):.2f}% - {mean_extent + 1.5 * np.std(percentages):.2f}%")
                st.markdown(f"- **High Risk:** Above {mean_extent + 1.5 * np.std(percentages):.2f}%")
        
        with col2:
            fig_gauge = go.Figure()
            
            if len(percentages) >= 2:
                current_val = percentages[-1]
                max_val = max(percentages) * 1.2
                
                fig_gauge.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=current_val,
                    delta={'reference': np.mean(percentages)},
                    title={'text': f"Water Extent - {months[-1]}"},
                    gauge={
                        'axis': {'range': [0, max_val]},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range': [0, mean_extent], 'color': "lightgray"},
                            {'range': [mean_extent, mean_extent + 0.5 * np.std(percentages)], 'color': "lightyellow"},
                            {'range': [mean_extent + 0.5 * np.std(percentages), mean_extent + 1.5 * np.std(percentages)], 'color': "lightsalmon"},
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': mean_extent + 1.5 * np.std(percentages)
                        }
                    }
                ))
                
                fig_gauge.update_layout(height=400)
                st.plotly_chart(fig_gauge, width='stretch')
        
        st.markdown("---")
        st.markdown("### Flood Risk Analysis")
        
        if risk_level == "High Risk":
            st.error("⚠️ **HIGH FLOOD RISK DETECTED!** The current water extent significantly exceeds historical averages. Immediate monitoring and preparedness measures are recommended.")
        elif risk_level == "Moderate Risk":
            st.warning("⚡ **Moderate flood risk.** Water levels are elevated compared to historical trends. Continue monitoring the situation closely.")
        elif risk_level == "Low Risk":
            st.info("📊 Water levels are slightly above average. Normal monitoring protocols should continue.")
        else:
            st.success("✅ Water levels are within normal range based on historical data.")
        
        st.markdown("#### Historical Context")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            increase_count = sum(1 for i in range(1, len(percentages)) if percentages[i] > percentages[i-1])
            st.metric("Months with Increase", f"{increase_count}/{len(percentages)-1}")
        
        with col2:
            if len(percentages) >= 3:
                trend = "Increasing" if percentages[-1] > percentages[-3] else "Decreasing"
                st.metric("3-Month Trend", trend)
        
        with col3:
            volatility = np.std(percentages) / np.mean(percentages) * 100
            st.metric("Volatility Index", f"{volatility:.1f}%")
    
    with tab5:
        st.subheader("Predictive Flood Modeling")
        st.markdown("Forecasting future water extent based on historical trends and seasonal patterns")
        
        if len(months) < 3:
            st.warning("Insufficient data for predictive modeling. Need at least 3 months of historical data.")
        else:
            percentages_array = np.array(percentages)
            months_numeric = np.arange(len(months))
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.markdown("#### Forecast Settings")
                forecast_months = st.slider(
                    "Months to Forecast",
                    min_value=1,
                    max_value=6,
                    value=3,
                    help="Number of future months to predict"
                )
                
                confidence_level = st.select_slider(
                    "Confidence Level",
                    options=[80, 90, 95],
                    value=90,
                    help="Confidence interval for predictions"
                )
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(months_numeric, percentages_array)
            
            future_months_numeric = np.arange(len(months), len(months) + forecast_months)
            future_predictions = slope * future_months_numeric + intercept
            
            z_score = {80: 1.28, 90: 1.645, 95: 1.96}[confidence_level]
            prediction_std = np.std(percentages_array - (slope * months_numeric + intercept))
            confidence_interval = z_score * prediction_std
            
            upper_bound = future_predictions + confidence_interval
            lower_bound = future_predictions - confidence_interval
            
            future_month_labels = [f"Month +{i+1}" for i in range(forecast_months)]
            
            with col1:
                fig_forecast = go.Figure()
                
                fig_forecast.add_trace(go.Scatter(
                    x=list(range(len(months))),
                    y=percentages,
                    mode='lines+markers',
                    name='Historical Data',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8)
                ))
                
                fig_forecast.add_trace(go.Scatter(
                    x=list(future_months_numeric),
                    y=future_predictions,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#ff7f0e', width=3, dash='dash'),
                    marker=dict(size=8, symbol='diamond')
                ))
                
                fig_forecast.add_trace(go.Scatter(
                    x=list(future_months_numeric) + list(future_months_numeric[::-1]),
                    y=list(upper_bound) + list(lower_bound[::-1]),
                    fill='toself',
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{confidence_level}% Confidence Interval',
                    showlegend=True
                ))
                
                mean_line = np.mean(percentages)
                fig_forecast.add_hline(
                    y=mean_line,
                    line_dash="dot",
                    line_color="green",
                    annotation_text=f"Historical Average: {mean_line:.2f}%"
                )
                
                fig_forecast.update_layout(
                    title='Water Extent Forecast',
                    xaxis_title='Time Period',
                    yaxis_title='Water Extent (%)',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_forecast, width='stretch')
            
            st.markdown("---")
            st.markdown("#### Forecast Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_prediction = np.mean(future_predictions)
                st.metric(
                    "Avg Forecast Extent",
                    f"{avg_prediction:.2f}%",
                    delta=f"{avg_prediction - np.mean(percentages):+.2f}% vs historical"
                )
            
            with col2:
                trend_direction = "Increasing" if slope > 0 else "Decreasing"
                trend_rate = abs(slope)
                st.metric(
                    "Trend Direction",
                    trend_direction,
                    delta=f"{trend_rate:.2f}% per month"
                )
            
            with col3:
                model_confidence = r_value ** 2
                st.metric(
                    "Model Confidence (R²)",
                    f"{model_confidence:.3f}",
                    delta="Good fit" if model_confidence > 0.7 else "Moderate fit"
                )
            
            with col4:
                max_forecast = np.max(future_predictions)
                forecast_risk = "High" if max_forecast > mean_line + 1.5 * np.std(percentages) else "Moderate" if max_forecast > mean_line + 0.5 * np.std(percentages) else "Low"
                risk_colors_map = {"High": "red", "Moderate": "orange", "Low": "green"}
                st.metric(
                    "Forecasted Risk",
                    forecast_risk
                )
            
            st.markdown("#### Detailed Forecast")
            
            forecast_df = pd.DataFrame({
                'Period': future_month_labels,
                'Predicted Extent (%)': [f"{p:.2f}" for p in future_predictions],
                'Lower Bound (%)': [f"{l:.2f}" for l in lower_bound],
                'Upper Bound (%)': [f"{u:.2f}" for u in upper_bound],
                'Risk Level': ['High' if p > mean_line + 1.5 * np.std(percentages) else 'Moderate' if p > mean_line + 0.5 * np.std(percentages) else 'Normal' for p in future_predictions]
            })
            
            st.dataframe(forecast_df, width='stretch', hide_index=True)
            
            if forecast_risk == "High":
                st.error("⚠️ **HIGH FLOOD RISK FORECASTED!** The predicted water extent suggests potential flooding in upcoming months. Immediate preparedness measures recommended.")
            elif forecast_risk == "Moderate":
                st.warning("⚡ **Moderate flood risk forecasted.** Water levels are predicted to be elevated. Enhanced monitoring is advised.")
            else:
                st.success("✅ Forecasted water levels appear to be within safe ranges. Continue normal monitoring protocols.")
            
            st.markdown("---")
            st.markdown("#### Seasonal Pattern Analysis")
            
            if len(months) >= 4:
                try:
                    month_names = []
                    for m in months:
                        try:
                            year, month = m.split('-')
                            month_num = int(month)
                            month_names.append(month_num)
                        except:
                            month_names.append(0)
                    
                    if len(set(month_names)) > 1:
                        seasonal_df = pd.DataFrame({
                            'Month': month_names,
                            'Water Extent (%)': percentages
                        })
                        
                        seasonal_avg = seasonal_df.groupby('Month')['Water Extent (%)'].mean().reset_index()
                        
                        fig_seasonal = px.bar(
                            seasonal_avg,
                            x='Month',
                            y='Water Extent (%)',
                            title='Average Water Extent by Month (Seasonal Pattern)',
                            labels={'Month': 'Month of Year'},
                            color='Water Extent (%)',
                            color_continuous_scale='Blues'
                        )
                        
                        fig_seasonal.update_layout(height=350)
                        st.plotly_chart(fig_seasonal, width='stretch')
                        
                        peak_month = seasonal_avg.loc[seasonal_avg['Water Extent (%)'].idxmax(), 'Month']
                        low_month = seasonal_avg.loc[seasonal_avg['Water Extent (%)'].idxmin(), 'Month']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"📈 **Peak water levels** typically occur in month **{int(peak_month)}**")
                        with col2:
                            st.info(f"📉 **Lowest water levels** typically occur in month **{int(low_month)}**")
                    else:
                        st.info("Insufficient data for seasonal pattern analysis. Need data across different months of the year.")
                except Exception as e:
                    st.info("Unable to perform seasonal analysis with current data format.")
            else:
                st.info("Need at least 4 months of data for seasonal pattern analysis.")

if __name__ == "__main__":
    main()
