import React, { useState } from "react"
import { Rect, Text, Transformer } from 'react-konva';

export interface RectProps {
  x: number,
  y: number,
  width: number,
  height: number,
  id: string,
  stroke: string,
  label: string,
}
export interface BBoxProps {
  rectProps: RectProps,
  onChange: any,
  onDelete: () => void,
  isSelected: boolean,
  onClick: any,
  scale: number,
  strokeWidth: number,
  fillOpacity: number,
}

const hexToRgba = (hex: string, alpha: number): string => {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `rgba(${r},${g},${b},${alpha})`
}

const BBox = (props: BBoxProps) => {
  const shapeRef = React.useRef<any>();
  const trRef = React.useRef<any>();
  const {
    rectProps, onChange, onDelete, isSelected, onClick, scale, strokeWidth, fillOpacity
  }: BBoxProps = props
  const [moving, setMoving] = useState(false);

  React.useEffect(() => {
    trRef.current?.nodes([shapeRef.current]);
    trRef.current?.getLayer().batchDraw();
  }, [isSelected]);

  return (
    <React.Fragment>
      <Rect
        onClick={onClick}
        onMouseDown={(e: any) => {
          if (e.evt.button === 1) {
            e.evt.preventDefault()
            onDelete()
          }
        }}
        ref={shapeRef}
        {...rectProps}
        stroke={''}
        x={rectProps.x * scale}
        y={rectProps.y * scale}
        width={rectProps.width * scale}
        height={rectProps.height * scale}
        fill={fillOpacity > 0 ? hexToRgba(rectProps.stroke, fillOpacity) : undefined}
        draggable={isSelected}
        strokeWidth={strokeWidth}
        onDragStart={() => { setMoving(true) }}
        onDragEnd={(e) => {
          onChange({
            ...rectProps,
            x: e.target.x() / scale,
            y: e.target.y() / scale,
          });
          setMoving(false)
        }}
        onTransformStart={() => { setMoving(true) }}
        onTransformEnd={(e) => {
          const node = shapeRef.current;
          const scaleX = node.scaleX();
          const scaleY = node.scaleY();
          setMoving(false)
          node.scaleX(1);
          node.scaleY(1);
          onChange({
            ...rectProps,
            x: node.x() / scale,
            y: node.y() / scale,
            width: Math.max(5, node.width() * scaleX / scale),
            height: Math.max(5, node.height() * scaleY / scale),
          });
        }}
      />
      {/* Texte rendu après le Rect pour rester lisible au-dessus du remplissage */}
      {moving || <Text text={rectProps.label} x={rectProps.x * scale + 5} y={rectProps.y * scale + 5} fontSize={15} fill={rectProps.stroke} />}
      <Transformer
        ref={trRef}
        resizeEnabled={isSelected}
        rotateEnabled={false}
        keepRatio={false}
        borderStroke={rectProps.stroke}
        borderStrokeWidth={strokeWidth}
        boundBoxFunc={(oldBox, newBox) => {
          if (newBox.width < 5 || newBox.height < 5) return oldBox;
          return newBox;
        }}
      />
    </React.Fragment>
  );
};

export default BBox;
