# Surface Water Extent Mapping Application

## Overview

This is a geospatial visualization application built with Streamlit that analyzes and displays surface water extent data from satellite imagery. The application processes GeoTIFF (.tif) files containing water extent measurements across different time periods and provides interactive visualizations including maps, charts, and statistical analysis. Users can explore temporal changes in water coverage through an intuitive web interface.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit - A Python-based web application framework chosen for rapid development of data science applications
- **Layout**: Wide layout configuration to maximize screen real estate for maps and visualizations
- **Visualization Libraries**:
  - Plotly (graph_objects and express) for interactive charts and graphs
  - Folium for interactive web maps with geographic overlays
  - Matplotlib for static plot generation with custom colormaps
  - streamlit-folium for embedding Folium maps in Streamlit

**Rationale**: Streamlit eliminates the need for separate frontend/backend development, allowing Python-based data processing and visualization in a single codebase. This architecture is ideal for data-focused applications where interactivity is needed but complex UI frameworks would be overkill.

### Data Processing Architecture
- **Geospatial Data Handling**: Rasterio library for reading GeoTIFF files, extracting raster data, geographic bounds, coordinate reference systems (CRS), and affine transformations
- **Data Caching**: `@st.cache_data` decorator on the `load_tif_files()` function to prevent redundant file I/O operations
- **File Naming Convention**: Files follow pattern `{prefix}_{month_str}_{suffix}.tif` where month_str is extracted for temporal organization
- **Data Structure**: Dictionary-based storage where keys are month strings and values contain:
  - Raw raster data arrays
  - Geographic bounds
  - CRS information
  - Affine transform matrices
  - File metadata

**Rationale**: Caching is critical for performance since GeoTIFF files can be large. The dictionary structure provides O(1) lookup by time period while maintaining all necessary geographic metadata for proper visualization.

### File Organization
- **Asset Directory**: `attached_assets/` directory contains all GeoTIFF source files
- **Entry Point**: `app.py` serves as the main application file
- **Legacy File**: `main.py` appears to be a placeholder/template file not used in the actual application

**Rationale**: Separating data assets from code allows for easy data updates without code changes and keeps the repository clean.

### Data Visualization Strategy
- **Multi-format Support**: Application supports both raster (image-based) and vector (point/polygon) visualizations
- **Color Mapping**: Custom matplotlib colormaps for categorical water extent data
- **Image Export**: PIL (Python Imaging Library) for converting matplotlib figures to image buffers for download functionality

**Alternatives Considered**: Could use pure JavaScript mapping libraries (Leaflet, Mapbox) but this would require a separate backend API. The Python-centric approach maintains simplicity.

**Pros**: 
- Unified codebase in Python
- Rapid prototyping and iteration
- Strong geospatial library ecosystem

**Cons**:
- Limited customization compared to React/Vue-based frontends
- Performance constraints for very large datasets (mitigated by caching)
- Less control over UI/UX details

## External Dependencies

### Python Libraries
- **streamlit**: Web application framework
- **rasterio**: GeoTIFF and raster data I/O
- **numpy**: Numerical array operations on raster data
- **pandas**: Tabular data manipulation and analysis
- **plotly**: Interactive visualization library
- **folium**: Interactive map generation
- **streamlit-folium**: Streamlit component for Folium integration
- **matplotlib**: Static plotting and color mapping
- **Pillow (PIL)**: Image processing and format conversion

### Data Format Requirements
- **Input Format**: GeoTIFF (.tif) files with embedded geographic metadata
- **Expected Metadata**: CRS, affine transform, and bounding box information must be present in files
- **File Naming**: Files must follow naming convention allowing month extraction from filename

### File System Dependencies
- **Asset Storage**: Application expects `attached_assets/` directory to exist and contain .tif files
- **No Database**: Application does not use a persistent database; all data is loaded from files on startup

**Note**: While no database is currently implemented, temporal data could benefit from a PostgreSQL database with PostGIS extension for advanced geospatial queries if the application scales to handle larger datasets or requires user-specific data persistence.