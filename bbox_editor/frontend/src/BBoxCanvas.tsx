import React, { useState, useEffect } from "react"
import { Layer, Rect, Stage, Image } from 'react-konva';
import BBox from './BBox'
import Konva from 'konva';

export interface BBoxCanvasLayerProps {
  rectangles: any[],
  selectedId: string | null,
  setSelectedId: any,
  setRectangles: any,
  setLabel: any,
  color_map: any,
  scale: number,
  label: string,
  image_size: number[],
  image: any,
  strokeWidth: number
}

const BBoxCanvas = (props: BBoxCanvasLayerProps) => {
  const {
    rectangles,
    selectedId,
    setSelectedId,
    setRectangles,
    setLabel,
    color_map,
    scale,
    label,
    image_size,
    image,
    strokeWidth
  }: BBoxCanvasLayerProps = props

  const [adding, setAdding] = useState<number[] | null>(null)

  // Clic sur la zone vide : démarrer un dessin si rien n'est sélectionné,
  // sinon désélectionner.
  const checkDeselect = (e: any) => {
    if (!(e.target instanceof Konva.Rect)) {
      if (selectedId === null) {
        const pointer = e.target.getStage().getPointerPosition()
        setAdding([pointer.x, pointer.y, pointer.x, pointer.y])
      } else {
        setSelectedId(null);
      }
    }
  };

  useEffect(() => {
    const rects = rectangles.slice();
    for (let i = 0; i < rects.length; i++) {
      if (rects[i].width < 0) {
        rects[i].width = rects[i].width * -1
        rects[i].x = rects[i].x - rects[i].width
        setRectangles(rects)
      }
      if (rects[i].height < 0) {
        rects[i].height = rects[i].height * -1
        rects[i].y = rects[i].y - rects[i].height
        setRectangles(rects)
      }
      if (rects[i].x < 0 || rects[i].y < 0) {
        rects[i].width = rects[i].width + Math.min(0, rects[i].x)
        rects[i].x = Math.max(0, rects[i].x)
        rects[i].height = rects[i].height + Math.min(0, rects[i].y)
        rects[i].y = Math.max(0, rects[i].y)
        setRectangles(rects)
      }
      if (rects[i].x + rects[i].width > image_size[0] || rects[i].y + rects[i].height > image_size[1]) {
        rects[i].width = Math.min(rects[i].width, image_size[0] - rects[i].x)
        rects[i].height = Math.min(rects[i].height, image_size[1] - rects[i].y)
        setRectangles(rects)
      }
      if (rects[i].width < 5 || rects[i].height < 5) {
        rects[i].width = 5
        rects[i].height = 5
      }
    }
  }, [rectangles, image_size])

  return (
    <Stage
      width={image_size[0] * scale}
      height={image_size[1] * scale}
      onMouseDown={checkDeselect}
      onContextMenu={(e) => { e.evt.preventDefault(); }}
      onMouseMove={(e: any) => {
        if (adding !== null) {
          const pointer = e.target.getStage().getPointerPosition()
          setAdding([adding[0], adding[1], pointer.x, pointer.y])
        }
      }}
      onMouseLeave={() => { setAdding(null) }}
      onMouseUp={() => {
        if (adding !== null) {
          const rects = rectangles.slice();
          const new_id = Date.now().toString()
          rects.push({
            x: adding[0] / scale,
            y: adding[1] / scale,
            width: (adding[2] - adding[0]) / scale,
            height: (adding[3] - adding[1]) / scale,
            label: label,
            stroke: color_map[label],
            id: new_id
          })
          setRectangles(rects);
          setAdding(null)
        }
      }}
    >
      <Layer>
        <Image image={image} scaleX={scale} scaleY={scale} />
      </Layer>
      <Layer>
        {rectangles.map((rect, i) => (
          <BBox
            key={i}
            rectProps={rect}
            scale={scale}
            strokeWidth={strokeWidth}
            isSelected={rect.id === selectedId}
            onClick={() => {
              setSelectedId(rect.id);
              const rects = rectangles.slice();
              const lastIndex = rects.length - 1;
              const lastItem = rects[lastIndex];
              rects[lastIndex] = rects[i];
              rects[i] = lastItem;
              setRectangles(rects);
              setLabel(rect.label)
            }}
            onDelete={() => {
              setRectangles(rectangles.filter((r) => r.id !== rect.id));
              setSelectedId(null);
            }}
            onChange={(newAttrs: any) => {
              const rects = rectangles.slice();
              rects[i] = newAttrs;
              setRectangles(rects);
            }}
          />
        ))}
        {adding !== null && (
          <Rect
            fill={color_map[label] + '4D'}
            x={adding[0]} y={adding[1]}
            width={adding[2] - adding[0]}
            height={adding[3] - adding[1]}
          />
        )}
      </Layer>
    </Stage>
  );
};

export default BBoxCanvas;
