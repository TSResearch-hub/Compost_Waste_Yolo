import React, { useState, useEffect, useRef } from "react"
import { Layer, Line, Rect, Stage, Image, Group } from 'react-konva';
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
  strokeWidth: number,
  opacity: number,
  brightness: number,
  contrast: number
}

const ZOOM_FACTOR = 1.12
const ZOOM_MIN    = 0.25
const ZOOM_MAX    = 12

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
    strokeWidth,
    opacity,
    brightness,
    contrast
  }: BBoxCanvasLayerProps = props

  const imageRef = useRef<Konva.Image | null>(null)

  useEffect(() => {
    const node = imageRef.current
    if (node && image) {
      node.cache()
      node.getLayer()?.batchDraw()
    }
  }, [image, brightness, contrast])

  const [adding,   setAdding]   = useState<number[] | null>(null)
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null)
  const [zoom,     setZoom]     = useState(1.0)
  const [stagePos, setStagePos] = useState({ x: 0, y: 0 })
  const [isPanning, setIsPanning] = useState(false)
  const panStart = useRef<{ mx: number; my: number; sx: number; sy: number } | null>(null)

  // Convertit des coordonnées viewport en coordonnées contenu du Stage
  const toStage = (stage: any, vx: number, vy: number) => ({
    x: (vx - stage.x()) / stage.scaleX(),
    y: (vy - stage.y()) / stage.scaleY(),
  })

  // Réinitialise zoom + position quand une nouvelle image est chargée
  useEffect(() => {
    setZoom(1.0)
    setStagePos({ x: 0, y: 0 })
  }, [image_size])

  // Clic gauche sur zone vide : démarrer dessin ou désélectionner
  const checkDeselect = (e: any) => {
    if (!(e.target instanceof Konva.Rect)) {
      if (selectedId === null) {
        const sc = toStage(e.target.getStage(), ...Object.values(e.target.getStage().getPointerPosition()) as [number, number])
        setAdding([sc.x, sc.y, sc.x, sc.y])
      } else {
        setSelectedId(null)
      }
    }
  }

  // Clamping des bboxes hors limites
  useEffect(() => {
    const rects = rectangles.slice()
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

  const W = image_size[0] * scale
  const H = image_size[1] * scale

  return (
    <Stage
      width={W}
      height={H}
      x={stagePos.x}
      y={stagePos.y}
      scaleX={zoom}
      scaleY={zoom}
      style={{ cursor: isPanning ? 'grabbing' : 'crosshair' }}

      // ── Zoom molette, centré sur le curseur ──────────────────────────────
      onWheel={(e: any) => {
        e.evt.preventDefault()
        const stage    = e.target.getStage()
        const oldZoom  = stage.scaleX()
        const pointer  = stage.getPointerPosition()
        const origin   = { x: (pointer.x - stage.x()) / oldZoom,
                           y: (pointer.y - stage.y()) / oldZoom }
        const newZoom  = e.evt.deltaY < 0
          ? Math.min(oldZoom * ZOOM_FACTOR, ZOOM_MAX)
          : Math.max(oldZoom / ZOOM_FACTOR, ZOOM_MIN)
        setZoom(newZoom)
        setStagePos({
          x: pointer.x - origin.x * newZoom,
          y: pointer.y - origin.y * newZoom,
        })
      }}

      // ── Double-clic : reset zoom ─────────────────────────────────────────
      onDblClick={() => {
        setZoom(1.0)
        setStagePos({ x: 0, y: 0 })
      }}

      // ── MouseDown : bouton droit = pan, gauche = dessin/deselect ─────────
      onMouseDown={(e: any) => {
        if (e.evt.button === 2) {
          e.evt.preventDefault()
          const p = e.target.getStage().getPointerPosition()
          panStart.current = { mx: p.x, my: p.y,
                               sx: e.target.getStage().x(),
                               sy: e.target.getStage().y() }
          setIsPanning(true)
          return
        }
        if (e.evt.button === 0) checkDeselect(e)
      }}

      onContextMenu={(e) => { e.evt.preventDefault() }}

      // ── MouseMove : pan ou mise à jour crosshair + rect en cours ─────────
      onMouseMove={(e: any) => {
        const stage   = e.target.getStage()
        const pointer = stage.getPointerPosition()

        if (panStart.current) {
          setStagePos({
            x: panStart.current.sx + (pointer.x - panStart.current.mx),
            y: panStart.current.sy + (pointer.y - panStart.current.my),
          })
          return
        }

        const sc = toStage(stage, pointer.x, pointer.y)
        setMousePos(sc)
        if (adding !== null) {
          setAdding([adding[0], adding[1], sc.x, sc.y])
        }
      }}

      onMouseLeave={() => {
        setAdding(null)
        setMousePos(null)
        panStart.current = null
        setIsPanning(false)
      }}

      // ── MouseUp : fin de pan ou création de bbox ─────────────────────────
      onMouseUp={(e: any) => {
        if (e.evt.button === 2) {
          panStart.current = null
          setIsPanning(false)
          return
        }
        if (adding !== null) {
          const rects  = rectangles.slice()
          rects.push({
            x:      adding[0] / scale,
            y:      adding[1] / scale,
            width:  (adding[2] - adding[0]) / scale,
            height: (adding[3] - adding[1]) / scale,
            label,
            stroke: color_map[label],
            id:     Date.now().toString()
          })
          setRectangles(rects)
          setAdding(null)
        }
      }}
    >
      {/* Couche image */}
      <Layer>
        <Image
          ref={imageRef}
          image={image}
          scaleX={scale}
          scaleY={scale}
          filters={[Konva.Filters.Brighten, Konva.Filters.Contrast] as any}
          brightness={brightness}
          contrast={contrast}
        />
      </Layer>

      {/* Couche bboxes + rect en cours de dessin */}
      <Layer>
        <Group opacity={opacity}>
        {rectangles.map((rect, i) => (
          <BBox
            key={i}
            rectProps={rect}
            scale={scale}
            strokeWidth={strokeWidth}
            isSelected={rect.id === selectedId}
            onClick={() => {
              setSelectedId(rect.id)
              const rects     = rectangles.slice()
              const lastIndex = rects.length - 1
              const lastItem  = rects[lastIndex]
              rects[lastIndex] = rects[i]
              rects[i]         = lastItem
              setRectangles(rects)
              setLabel(rect.label)
            }}
            onDelete={() => {
              setRectangles(rectangles.filter((r) => r.id !== rect.id))
              setSelectedId(null)
            }}
            onChange={(newAttrs: any) => {
              const rects = rectangles.slice()
              rects[i]    = newAttrs
              setRectangles(rects)
            }}
          />
        ))}
        </Group>
        {adding !== null && (
          <Rect
            fill={color_map[label] + '4D'}
            x={adding[0]} y={adding[1]}
            width={adding[2]  - adding[0]}
            height={adding[3] - adding[1]}
          />
        )}
      </Layer>

      {/* Couche crosshair — non-interactive, toujours au-dessus */}
      {mousePos !== null && (
        <Layer listening={false}>
          <Line points={[mousePos.x, 0, mousePos.x, H]}
                stroke="rgba(0,0,0,0.45)" strokeWidth={3 / zoom} dash={[6 / zoom, 6 / zoom]} />
          <Line points={[mousePos.x, 0, mousePos.x, H]}
                stroke="rgba(255,255,255,0.9)" strokeWidth={1 / zoom} dash={[6 / zoom, 6 / zoom]} />
          <Line points={[0, mousePos.y, W, mousePos.y]}
                stroke="rgba(0,0,0,0.45)" strokeWidth={3 / zoom} dash={[6 / zoom, 6 / zoom]} />
          <Line points={[0, mousePos.y, W, mousePos.y]}
                stroke="rgba(255,255,255,0.9)" strokeWidth={1 / zoom} dash={[6 / zoom, 6 / zoom]} />
        </Layer>
      )}
    </Stage>
  )
}

export default BBoxCanvas
