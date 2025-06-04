import streamlit as st
import ezdxf
from shapely.geometry import box, Polygon, LineString, GeometryCollection, Point
from shapely.ops import unary_union
from shapely.affinity import translate, rotate
from rectpack import newPacker
import tempfile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from io import BytesIO
import numpy as np
import os
from typing import List, Tuple, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="DXF Smart Layout", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def group_connected_segments(line_segments: List[Tuple]) -> List[List[Tuple]]:
    """Группирует связанные отрезки в отдельные фигуры."""
    if not line_segments:
        return []
    
    # Создаем словарь точек для быстрого поиска связей
    point_to_segments = {}
    for i, (start, end) in enumerate(line_segments):
        for point in [start, end]:
            if point not in point_to_segments:
                point_to_segments[point] = []
            point_to_segments[point].append(i)
    
    visited = set()
    groups = []
    
    for i, segment in enumerate(line_segments):
        if i in visited:
            continue
            
        # Начинаем новую группу
        current_group = []
        to_visit = [i]
        
        while to_visit:
            current_idx = to_visit.pop()
            if current_idx in visited:
                continue
                
            visited.add(current_idx)
            current_segment = line_segments[current_idx]
            current_group.append(current_segment)
            
            # Ищем все связанные сегменты
            for point in [current_segment[0], current_segment[1]]:
                if point in point_to_segments:
                    for connected_idx in point_to_segments[point]:
                        if connected_idx not in visited:
                            to_visit.append(connected_idx)
        
        if current_group:
            groups.append(current_group)
    
    return groups

def detect_shape_type(segments: List[Tuple]) -> str:
    """Определяет тип фигуры по сегментам."""
    if len(segments) < 3:
        return "line"
    
    # Собираем все точки
    all_points = []
    for start, end in segments:
        all_points.extend([start, end])
    
    unique_points = list(set(all_points))
    
    # Проверяем на круг/дугу
    if len(segments) > 8:  # Много сегментов - возможно круг
        # Вычисляем центр и радиус
        x_coords = [p[0] for p in unique_points]
        y_coords = [p[1] for p in unique_points]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        # Проверяем, все ли точки на одинаковом расстоянии от центра
        distances = [((p[0] - center_x)**2 + (p[1] - center_y)**2)**0.5 for p in unique_points]
        avg_dist = sum(distances) / len(distances)
        variance = sum((d - avg_dist)**2 for d in distances) / len(distances)
        
        if variance < avg_dist * 0.1:  # Малая вариация = круг
            return "circle"
    
    # Проверяем замкнутость
    start_points = [seg[0] for seg in segments]
    end_points = [seg[1] for seg in segments]
    
    # Если первая точка первого сегмента совпадает с последней точкой последнего
    if len(segments) > 2 and (start_points[0] in end_points or end_points[-1] in start_points):
        return "polygon"
    
    return "polyline"

def create_shape_geometry(segments: List[Tuple], shape_type: str) -> Polygon:
    """Создает геометрию фигуры в зависимости от типа."""
    if not segments:
        return None
    
    if shape_type == "circle":
        # Для круга вычисляем центр и радиус
        all_points = []
        for start, end in segments:
            all_points.extend([start, end])
        
        unique_points = list(set(all_points))
        x_coords = [p[0] for p in unique_points]
        y_coords = [p[1] for p in unique_points]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        distances = [((p[0] - center_x)**2 + (p[1] - center_y)**2)**0.5 for p in unique_points]
        radius = sum(distances) / len(distances)
        
        return Point(center_x, center_y).buffer(radius * 0.95)  # Немного меньше для безопасности
    
    elif shape_type in ["polygon", "polyline"]:
        try:
            # Пытаемся создать полигон из точек
            all_points = []
            for start, end in segments:
                all_points.extend([start, end])
            
            unique_points = list(set(all_points))
            if len(unique_points) >= 3:
                from shapely.geometry import MultiPoint
                return MultiPoint(unique_points).convex_hull
        except:
            pass
    
    # Fallback: bounding box
    all_points = []
    for start, end in segments:
        all_points.extend([start, end])
    
    if all_points:
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]
        return box(min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    return None

@st.cache_data
def get_model_geometry(dxf_path: str, _cache_key: str = None) -> Tuple[List[Dict], float, float, Tuple, float, float]:
    """Extract geometry and calculate bounds from DXF file with shape grouping."""
    try:
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        all_line_segments = []
        individual_shapes = []  # Для кругов и других отдельных фигур

        for entity in msp:
            try:
                if entity.dxftype() == 'LINE':
                    start = (entity.dxf.start.x, entity.dxf.start.y)
                    end = (entity.dxf.end.x, entity.dxf.end.y)
                    all_line_segments.append((start, end))
                
                elif entity.dxftype() in ('LWPOLYLINE', 'POLYLINE'):
                    points = [tuple(p[:2]) for p in entity.get_points()]
                    if len(points) >= 2:
                        segments = []
                        for i in range(len(points) - 1):
                            segments.append((points[i], points[i+1]))
                        
                        # Add closing segment if polyline is closed
                        if hasattr(entity.dxf, 'flags') and entity.dxf.flags & 1:
                            segments.append((points[-1], points[0]))
                        
                        all_line_segments.extend(segments)
                
                elif entity.dxftype() == 'CIRCLE':
                    center = (entity.dxf.center.x, entity.dxf.center.y)
                    radius = entity.dxf.radius
                    
                    # Создаем отдельную фигуру для круга
                    individual_shapes.append({
                        'type': 'circle',
                        'geometry': Point(center).buffer(radius),
                        'segments': [],  # Круг - цельная фигура
                        'bounds': (center[0] - radius, center[1] - radius, 
                                 center[0] + radius, center[1] + radius)
                    })
                
                elif entity.dxftype() == 'ARC':
                    center = (entity.dxf.center.x, entity.dxf.center.y)
                    radius = entity.dxf.radius
                    start_angle = np.radians(entity.dxf.start_angle)
                    end_angle = np.radians(entity.dxf.end_angle)
                    
                    # Аппроксимируем дугу сегментами
                    num_segments = max(8, int(abs(end_angle - start_angle) * 8 / np.pi))
                    arc_segments = []
                    for i in range(num_segments):
                        angle1 = start_angle + (end_angle - start_angle) * i / num_segments
                        angle2 = start_angle + (end_angle - start_angle) * (i + 1) / num_segments
                        p1 = (center[0] + radius * np.cos(angle1), center[1] + radius * np.sin(angle1))
                        p2 = (center[0] + radius * np.cos(angle2), center[1] + radius * np.sin(angle2))
                        arc_segments.append((p1, p2))
                    
                    all_line_segments.extend(arc_segments)

            except Exception as e:
                logger.warning(f"Skipping entity {entity.dxftype()}: {e}")
                continue

        # Группируем линейные сегменты в фигуры
        grouped_segments = group_connected_segments(all_line_segments)
        
        shapes = []
        all_geometries = []
        
        # Обрабатываем группы сегментов
        for i, segments in enumerate(grouped_segments):
            shape_type = detect_shape_type(segments)
            geometry = create_shape_geometry(segments, shape_type)
            
            if geometry:
                bounds = geometry.bounds
                shapes.append({
                    'id': f'group_{i}',
                    'type': shape_type,
                    'geometry': geometry,
                    'segments': segments,
                    'bounds': bounds
                })
                all_geometries.append(geometry)
        
        # Добавляем индивидуальные фигуры (круги и т.д.)
        for shape in individual_shapes:
            shapes.append(shape)
            all_geometries.append(shape['geometry'])

        if not all_geometries:
            raise ValueError("DXF file contains no supported geometry")

        # Вычисляем общие границы
        union_geometry = unary_union(all_geometries)
        bounds = union_geometry.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        offset_x = -bounds[0]
        offset_y = -bounds[1]

        return shapes, width, height, bounds, offset_x, offset_y

    except Exception as e:
        raise ValueError(f"Error reading DXF file: {str(e)}")

def check_shapes_fit(shapes: List[Dict], sheet_polygon: Polygon, 
                    placement_x: float, placement_y: float, 
                    offset_x: float, offset_y: float, rotated: bool = False,
                    margin: float = 1.0) -> bool:
    """Check if all shapes fit within the sheet bounds."""
    try:
        sheet_with_margin = box(margin, margin, 
                               sheet_polygon.bounds[2] - margin, 
                               sheet_polygon.bounds[3] - margin)
        
        for shape in shapes:
            geometry = shape['geometry']
            
            if rotated:
                # Rotate around shape center
                bounds = geometry.bounds
                center_x = (bounds[0] + bounds[2]) / 2
                center_y = (bounds[1] + bounds[3]) / 2
                geometry = rotate(geometry, 90, origin=(center_x, center_y))
            
            # Apply translation
            final_geometry = translate(geometry, 
                                     xoff=placement_x + offset_x, 
                                     yoff=placement_y + offset_y)
            
            # Check if this shape fits
            if not sheet_with_margin.contains(final_geometry):
                return False
        
        return True
    
    except Exception as e:
        logger.warning(f"Shapes fit check failed: {e}")
        return False

def optimize_placement(model_width: float, model_height: float, sheet_width: float, 
                      sheet_height: float, spacing: float = 0.0, 
                      allow_rotation: bool = False, max_pieces: int = 1000,
                      shapes: List[Dict] = None, offset_x: float = 0, offset_y: float = 0) -> Tuple[int, List]:
    """Управляемая укладка: размещение по строкам и столбцам с валидацией фигур."""

    sheet_polygon = box(0, 0, sheet_width, sheet_height)

    best_count = 0
    best_placement = []

    orientations = [(model_width, model_height, False)]
    if allow_rotation and model_width != model_height:
        orientations.append((model_height, model_width, True))

    for width, height, is_rotated in orientations:
        adjusted_w = width + spacing
        adjusted_h = height + spacing

        cols = int(sheet_width // adjusted_w)
        rows = int(sheet_height // adjusted_h)
        theoretical_max = min(rows * cols, max_pieces)

        valid_placements = []
        count = 0

        for row in range(rows):
            for col in range(cols):
                if count >= theoretical_max:
                    break
                x = col * adjusted_w
                y = row * adjusted_h
                placement_x = x + spacing / 2
                placement_y = y + spacing / 2

                if shapes:
                    if check_shapes_fit(shapes, sheet_polygon, 
                                         placement_x, placement_y, 
                                         offset_x, offset_y, is_rotated):
                        valid_placements.append((x, y, width + spacing, height + spacing, None, count, is_rotated))
                        count += 1
                else:
                    if (placement_x + width + offset_x <= sheet_width and 
                        placement_y + height + offset_y <= sheet_height):
                        valid_placements.append((x, y, width + spacing, height + spacing, None, count, is_rotated))
                        count += 1

        if count > best_count:
            best_count = count
            best_placement = valid_placements

    return best_count, best_placement

def create_visualization(shapes: List[Dict], placed_rects: List, 
                        sheet_width: float, sheet_height: float, 
                        model_bounds: Tuple, offset_x: float, offset_y: float,
                        spacing: float, model_width: float, model_height: float) -> BytesIO:
    """Create visualization showing grouped shapes."""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set up the plot
    margin = max(sheet_width, sheet_height) * 0.05
    ax.set_xlim(-margin, sheet_width + margin)
    ax.set_ylim(-margin, sheet_height + margin)
    ax.set_aspect('equal')
    
    # Draw sheet background
    sheet_rect = Rectangle((0, 0), sheet_width, sheet_height, 
                          edgecolor='black', facecolor='lightgray', 
                          linewidth=2, alpha=0.3)
    ax.add_patch(sheet_rect)
    
    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(placed_rects)))
    
    # Draw each placed model
    for i, placement in enumerate(placed_rects):
        x, y, w, h = placement[0], placement[1], placement[2], placement[3]
        rotated = len(placement) > 6 and placement[6]
        
        # Calculate model position
        model_x = x + spacing / 2 + offset_x
        model_y = y + spacing / 2 + offset_y
        
        # Draw bounding rectangle
        rect = Rectangle((x, y), w, h, 
                        edgecolor='blue', facecolor=colors[i % len(colors)], 
                        alpha=0.3, linewidth=1)
        ax.add_patch(rect)
        
        # Draw each shape in the group
        for j, shape in enumerate(shapes):
            if shape['type'] == 'circle':
                # Рисуем круг
                center = shape['geometry'].centroid
                radius = shape['geometry'].area ** 0.5 / np.pi ** 0.5
                
                if rotated:
                    # Для круга поворот не меняет форму
                    circle_x = center.x + model_x
                    circle_y = center.y + model_y
                else:
                    circle_x = center.x + model_x
                    circle_y = center.y + model_y
                
                circle = plt.Circle((circle_x, circle_y), radius, 
                                  fill=False, edgecolor='red', linewidth=1.5)
                ax.add_patch(circle)
            
            else:
                # Рисуем сегменты фигуры
                color = 'red' if j == 0 else 'darkred'  # Первая фигура ярче
                for start, end in shape['segments']:
                    if rotated:
                        # Rotate points around model center
                        cx = (model_bounds[0] + model_bounds[2]) / 2
                        cy = (model_bounds[1] + model_bounds[3]) / 2
                        start_rot = (-(start[1] - cy) + cx, start[0] - cx + cy)
                        end_rot = (-(end[1] - cy) + cx, end[0] - cx + cy)
                        
                        ax.plot([start_rot[0] + model_x, end_rot[0] + model_x],
                               [start_rot[1] + model_y, end_rot[1] + model_y],
                               color=color, linewidth=1.2, alpha=0.8)
                    else:
                        ax.plot([start[0] + model_x, end[0] + model_x],
                               [start[1] + model_y, end[1] + model_y],
                               color=color, linewidth=1.2, alpha=0.8)
        
        # Add piece number
        ax.text(x + w/2, y + h/2, str(i + 1), 
               ha='center', va='center', fontsize=12, 
               fontweight='bold', color='black',
               bbox=dict(boxstyle='circle', facecolor='white', alpha=0.9))
    
    # Add shape count info
    shape_types = {}
    for shape in shapes:
        shape_type = shape['type']
        shape_types[shape_type] = shape_types.get(shape_type, 0) + 1
    
    info_text = f"Фигур в модели: {', '.join([f'{count} {type}' for type, count in shape_types.items()])}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add grid and labels
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_title(f'Оптимизированная раскладка: {len(placed_rects)} деталей на листе {sheet_width}×{sheet_height}мм', 
                fontsize=14, fontweight='bold')
    
    # Invert Y-axis to match DXF coordinate system
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return buf

def calculate_material_efficiency(placed_count: int, model_width: float, model_height: float,
                                sheet_width: float, sheet_height: float) -> dict:
    """Calculate material usage efficiency."""
    model_area = model_width * model_height
    total_model_area = placed_count * model_area
    sheet_area = sheet_width * sheet_height
    efficiency = (total_model_area / sheet_area) * 100
    waste_area = sheet_area - total_model_area
    
    return {
        'efficiency': efficiency,
        'waste_area': waste_area,
        'total_model_area': total_model_area,
        'sheet_area': sheet_area
    }

def generate_cut_list(placed_rects: List, sheet_width: float, sheet_height: float, 
                     shapes: List[Dict], spacing: float) -> str:
    """Generate detailed cut list."""
    cut_list = []
    cut_list.append("=== КАРТА РАСКРОЯ ===")
    cut_list.append(f"Лист: {sheet_width} × {sheet_height} мм")
    cut_list.append(f"Всего деталей: {len(placed_rects)}")
    
    # Shape types summary
    shape_types = {}
    for shape in shapes:
        shape_type = shape['type']
        shape_types[shape_type] = shape_types.get(shape_type, 0) + 1
    
    cut_list.append(f"Типы фигур в детали: {', '.join([f'{count} {type}' for type, count in shape_types.items()])}")
    cut_list.append("")
    
    # Detail positions
    for i, placement in enumerate(placed_rects):
        x, y, w, h = placement[0], placement[1], placement[2], placement[3]
        rotated = len(placement) > 6 and placement[6]
        rotation_text = " (повернуто 90°)" if rotated else ""
        cut_list.append(f"Деталь {i+1}: Позиция ({x:.1f}, {y:.1f}) мм, Размер {w:.1f}×{h:.1f} мм{rotation_text}")
    
    cut_list.append("")
    cut_list.append("=== ТЕХНИЧЕСКИЕ ДАННЫЕ ===")
    cut_list.append(f"Отступ между деталями: {spacing} мм")
    cut_list.append(f"Повернутых деталей: {sum(1 for p in placed_rects if len(p) > 6 and p[6])}")
    
    return "\n".join(cut_list)

# Streamlit UI
st.title("🔧 DXF Smart Layout Optimizer (с группировкой фигур)")
st.markdown("Загрузите DXF файл для оптимизации раскладки деталей на листовом материале с учетом группировки связанных элементов.")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Sheet dimensions
    st.subheader("Размеры листа")
    sheet_width = st.number_input("Ширина (мм)", min_value=10.0, value=1000.0, step=10.0)
    sheet_height = st.number_input("Высота (мм)", min_value=10.0, value=500.0, step=10.0)
    
    # Layout options
    st.subheader("Параметры раскладки")
    spacing = st.number_input("Отступ (мм)", min_value=0.0, value=2.0, step=0.5,
                             help="Минимальное расстояние между деталями")
    allow_rotation = st.checkbox("Разрешить поворот на 90°", value=True,
                               help="Разрешить поворот деталей для лучшего размещения")
    max_pieces = st.slider("Максимум деталей", min_value=1, max_value=500, value=100,
                          help="Ограничить расчет для производительности")

# Main content
uploaded_file = st.file_uploader("📁 Загрузить DXF файл", type=["dxf"])

if uploaded_file:
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Create cache key
    cache_key = f"{sheet_width}x{sheet_height}_{spacing}_{allow_rotation}_{max_pieces}"
    
    try:
        # Process DXF file
        with st.spinner("Обработка DXF файла и группировка фигур..."):
            shapes, model_w, model_h, model_bounds, offset_x, offset_y = get_model_geometry(tmp_path, cache_key)
        
        # Display shape analysis
        st.subheader("📊 Анализ фигур в модели")
        shape_types = {}
        for shape in shapes:
            shape_type = shape['type']
            shape_types[shape_type] = shape_types.get(shape_type, 0) + 1
        
        cols = st.columns(len(shape_types) + 2)
        with cols[0]:
            st.metric("Ширина модели", f"{model_w:.1f} мм")
        with cols[1]:
            st.metric("Высота модели", f"{model_h:.1f} мм")
        
        for i, (shape_type, count) in enumerate(shape_types.items()):
            with cols[i + 2]:
                st.metric(f"Фигур типа '{shape_type}'", count)
        
        # Show detailed shape info
        with st.expander("🔍 Детальная информация о фигурах"):
            for i, shape in enumerate(shapes):
                st.write(f"**Фигура {i+1}:** {shape['type']}")
                st.write(f"  - Границы: {[f'{x:.1f}' for x in shape['bounds']]}")
                if shape['segments']:
                    st.write(f"  - Сегментов: {len(shape['segments'])}")
                else:
                    st.write(f"  - Цельная фигура (например, круг)")
        
        # Check if model fits
        if (model_w > sheet_width or model_h > sheet_height) and \
           (not allow_rotation or (model_h > sheet_width or model_w > sheet_height)):
            st.error("❌ Модель слишком велика для листа в любой ориентации!")
            st.stop()
        
        # Calculate optimal placement
        with st.spinner("Оптимизация раскладки с проверкой фигур..."):
            placed_count, placed_rects = optimize_placement(
                model_w, model_h, sheet_width, sheet_height, 
                spacing, allow_rotation, max_pieces,
                shapes, offset_x, offset_y
            )
        
        if not placed_rects:
            st.error("❌ Невозможно разместить ни одной детали на листе!")
            st.stop()
        
        # Display results
        if placed_count > 0:
            st.success(f"✅ Успешно размещено **{placed_count}** полных деталей!")
            
            # Additional info for complex shapes
            if any(shape['type'] == 'circle' for shape in shapes):
                st.info("🔄 **Обнаружены круглые фигуры** - используется точная проверка размещения")
        
        # Calculate efficiency
        efficiency_data = calculate_material_efficiency(
            placed_count, model_w, model_h, sheet_width, sheet_height
        )
        
        # Metrics display
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Размещено деталей", placed_count)
        with col2:
            st.metric("Эффективность материала", f"{efficiency_data['efficiency']:.1f}%")
        with col3:
            st.metric("Отходы", f"{efficiency_data['waste_area']:.0f} мм²")
        with col4:
            rotation_count = sum(1 for p in placed_rects if len(p) > 6 and p[6])
            st.metric("Повернутых деталей", rotation_count)
        
        # Create and display visualization
        with st.spinner("Создание визуализации..."):
            viz_buffer = create_visualization(
                shapes, placed_rects, sheet_width, sheet_height,
                model_bounds, offset_x, offset_y, spacing, model_w, model_h
            )
        
        st.image(viz_buffer, caption="Оптимизированная раскладка", use_column_width=True)
        
        # Detailed breakdown
        with st.expander("📊 Подробный анализ"):
            st.write(f"**Размеры листа:** {sheet_width} × {sheet_height} мм")
            st.write(f"**Размер модели:** {model_w:.1f} × {model_h:.1f} мм")
            st.write(f"**Эффективность использования материала:** {efficiency_data['efficiency']:.2f}%")
            st.write(f"**Площадь отходов:** {efficiency_data['waste_area']:.2f} мм²")
            st.write(f"**Площадь всех моделей:** {efficiency_data['total_model_area']:.2f} мм²")
            st.write(f"**Площадь листа:** {efficiency_data['sheet_area']:.2f} мм²")

        # Export cut list
        with st.expander("📝 Список резки / отчет"):
            cut_report = generate_cut_list(placed_rects, sheet_width, sheet_height, shapes, spacing)
            st.text_area("Cut list", cut_report, height=300)
            st.download_button("Скачать список резки", 
							   cut_report, file_name="cut_list.txt", 
							   mime="text/plain", key="cut_list_download")
            
        # Additional output options
        with st.expander("📤 Экспортировать раскладку как изображение"):
            st.download_button(
                label="Скачать как PNG",
                data=viz_buffer,
                file_name="layout_preview.png",
                mime="image/png",
                key="download_layout_png"
            )

        st.success("✅ Расчет и визуализация завершены!")

    except Exception as e:
        st.error(f"Произошла ошибка: {str(e)}")
        logger.exception("Unhandled exception during layout process")
