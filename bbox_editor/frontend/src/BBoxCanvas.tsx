import React, { useState, useEffect, useRef } from "react"
import { Layer, Line, Rect, Stage, Image, Group } from 'react-konva';
import BBox from './BBox'
import Konva from 'konva';

export interface BBoxCanvasLayerProps {
  rectangles: any[],
  selectedId: string | null,
  setSelectedId: any,
  setRectangles: any,        // setter brut : sélection / réordonnancement z-order (hors historique)
  commitRectangles: any,     // setter avec historique : toute modification réelle des bboxes
  setLabel: any,
  color_map: any,
  scale: number,
  label: string,
  image_size: number[],
  image: any,
  strokeWidth: number,
  opacity: number,
  brightness: number,
  contrast: number,
  fillOpacity: number,
}

const ZOOM_FACTOR = 1.12
const ZOOM_MIN    = 0.25
const ZOOM_MAX    = 12
// En dessous de ce déplacement (px écran), un clic/tap sur zone vide est ignoré
// au lieu de créer une bbox minuscule accidentelle.
const MIN_DRAW_PX = 4

const BBoxCanvas = (props: BBoxCanvasLayerProps) => {
  const {
    rectangles,
    selectedId,
    setSelectedId,
    setRectangles,
    commitRectangles,
    setLabel,
    color_map,
    scale,
    label,
    image_size,
    image,
    strokeWidth,
    opacity,
    brightness,
    contrast,
    fillOpacity,
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
  // État du geste 2 doigts (pinch-zoom + pan) : distance/centre/zoom/position initiaux
  const pinchStart = useRef<{ dist: number; center: { x: number; y: number }; zoom: number; sx: number; sy: number } | null>(null)

  // Convertit des coordonnées viewport en coordonnées contenu du Stage
  const toStage = (stage: any, vx: number, vy: number) => ({
    x: (vx - stage.x()) / stage.scaleX(),
    y: (vy - stage.y()) / stage.scaleY(),
  })

  // Positions des doigts relatives au conteneur du Stage
  const touchPoints = (stage: any, touches: TouchList) => {
    const rect = stage.container().getBoundingClientRect()
    return Array.from(touches).map((t: any) => ({
      x: t.clientX - rect.left,
      y: t.clientY - rect.top,
    }))
  }

  // Normalise une bbox : largeur/hauteur positives, bornée à l'image, taille min 5px
  const clampRect = (r: any) => {
    let { x, y, width, height } = r
    if (width < 0)  { x += width;  width  = -width }
    if (height < 0) { y += height; height = -height }
    width  += Math.min(0, x); x = Math.max(0, x)
    height += Math.min(0, y); y = Math.max(0, y)
    width  = Math.max(5, Math.min(width,  image_size[0] - x))
    height = Math.max(5, Math.min(height, image_size[1] - y))
    x = Math.min(Math.max(0, x), Math.max(0, image_size[0] - width))
    y = Math.min(Math.max(0, y), Math.max(0, image_size[1] - height))
    return { ...r, x, y, width, height }
  }

  // Réinitialise zoom + position quand une nouvelle image est chargée
  useEffect(() => {
    setZoom(1.0)
    setStagePos({ x: 0, y: 0 })
  }, [image_size])

  // Clic/tap sur zone vide : démarrer dessin ou désélectionner
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

  // Fin de dessin (souris ou tactile) : crée la bbox si le geste est assez grand
  const finishAdding = () => {
    if (adding === null) return
    const dw = Math.abs(adding[2] - adding[0])
    const dh = Math.abs(adding[3] - adding[1])
    // Geste trop petit (simple clic/tap) → pas de bbox fantôme
    if (dw * zoom < MIN_DRAW_PX && dh * zoom < MIN_DRAW_PX) {
      setAdding(null)
      return
    }
    const newRect = clampRect({
      x:      adding[0] / scale,
      y:      adding[1] / scale,
      width:  (adding[2] - adding[0]) / scale,
      height: (adding[3] - adding[1]) / scale,
      label,
      stroke: color_map[label],
      id:     Date.now().toString()
    })
    commitRectangles([...rectangles, newRect])
    setAdding(null)
  }

  const applyZoomAt = (pointer: { x: number; y: number }, oldZoom: number, newZoom: number, sx: number, sy: number) => {
    const origin = { x: (pointer.x - sx) / oldZoom, y: (pointer.y - sy) / oldZoom }
    setZoom(newZoom)
    setStagePos({
      x: pointer.x - origin.x * newZoom,
      y: pointer.y - origin.y * newZoom,
    })
  }

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
      // touchAction none : le navigateur ne scrolle/zoome jamais la page
      // pendant qu'on manipule le canvas au doigt
      style={{ cursor: isPanning ? 'grabbing' : 'crosshair', touchAction: 'none' }}

      // ── Zoom molette, centré sur le curseur ──────────────────────────────
      onWheel={(e: any) => {
        e.evt.preventDefault()
        const stage    = e.target.getStage()
        const oldZoom  = stage.scaleX()
        const pointer  = stage.getPointerPosition()
        const newZoom  = e.evt.deltaY < 0
          ? Math.min(oldZoom * ZOOM_FACTOR, ZOOM_MAX)
          : Math.max(oldZoom / ZOOM_FACTOR, ZOOM_MIN)
        applyZoomAt(pointer, oldZoom, newZoom, stage.x(), stage.y())
      }}

      // ── Double-clic / double-tap : reset zoom ────────────────────────────
      onDblClick={() => {
        setZoom(1.0)
        setStagePos({ x: 0, y: 0 })
      }}
      onDblTap={() => {
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
        finishAdding()
      }}

      // ── TACTILE : 1 doigt = dessin/sélection · 2 doigts = zoom + pan ─────
      onTouchStart={(e: any) => {
        e.evt.preventDefault()
        const stage   = e.target.getStage()
        const touches = e.evt.touches

        if (touches.length === 2) {
          // Un 2e doigt posé annule le dessin en cours et démarre le pinch
          setAdding(null)
          const [p1, p2] = touchPoints(stage, touches)
          pinchStart.current = {
            dist:   Math.hypot(p2.x - p1.x, p2.y - p1.y),
            center: { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 },
            zoom:   stage.scaleX(),
            sx:     stage.x(),
            sy:     stage.y(),
          }
          return
        }
        if (touches.length === 1 && pinchStart.current === null) {
          checkDeselect(e)
        }
      }}

      onTouchMove={(e: any) => {
        e.evt.preventDefault()
        const stage   = e.target.getStage()
        const touches = e.evt.touches

        if (touches.length === 2 && pinchStart.current) {
          const [p1, p2] = touchPoints(stage, touches)
          const start   = pinchStart.current
          const dist    = Math.hypot(p2.x - p1.x, p2.y - p1.y)
          const center  = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 }
          const newZoom = Math.min(Math.max(start.zoom * dist / start.dist, ZOOM_MIN), ZOOM_MAX)
          // Zoom autour du centre initial du geste + pan si le centre se déplace
          const origin  = { x: (start.center.x - start.sx) / start.zoom,
                            y: (start.center.y - start.sy) / start.zoom }
          setZoom(newZoom)
          setStagePos({
            x: center.x - origin.x * newZoom,
            y: center.y - origin.y * newZoom,
          })
          return
        }

        if (adding !== null && touches.length === 1) {
          const pointer = stage.getPointerPosition()
          if (pointer) {
            const sc = toStage(stage, pointer.x, pointer.y)
            setAdding([adding[0], adding[1], sc.x, sc.y])
          }
        }
      }}

      onTouchEnd={(e: any) => {
        if (e.evt.touches.length < 2) pinchStart.current = null
        if (e.evt.touches.length === 0) finishAdding()
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
            fillOpacity={fillOpacity}
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
              commitRectangles(rectangles.filter((r) => r.id !== rect.id))
              setSelectedId(null)
            }}
            onChange={(newAttrs: any) => {
              const rects = rectangles.slice()
              rects[i]    = clampRect(newAttrs)
              commitRectangles(rects)
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
