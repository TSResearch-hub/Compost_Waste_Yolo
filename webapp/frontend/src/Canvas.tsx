/** Scène Konva — reprise de la logique de bbox_editor/BBoxCanvas.tsx :
 * zoom molette centré curseur, pan clic droit, dessin au clic-glisser sur
 * zone vide (quand rien n'est sélectionné), tactile 1 doigt = dessin /
 * 2 doigts = pinch-zoom + pan, croix de visée, filtres lumière/contraste. */
import Konva from "konva"
import { useEffect, useRef, useState } from "react"
import { Group, Image as KImage, Layer, Line, Rect, Stage } from "react-konva"

import Box, { type Geometrie } from "./Box"
import { couleurClasse, type BoxItem } from "./boxes"

const ZOOM_FACTEUR = 1.12
const ZOOM_MIN = 0.25
const ZOOM_MAX = 12
// En dessous de ce déplacement (px écran), un clic/tap sur zone vide est
// ignoré au lieu de créer une bbox minuscule accidentelle
const MIN_DESSIN_PX = 4

interface CanvasProps {
  boxes: BoxItem[]
  classes: string[]
  montrerRejetees: boolean
  selectedKey: string | null
  classeActive: number
  image: HTMLImageElement | undefined
  imageSize: [number, number] // dimensions du fichier annoté
  scale: number
  strokeWidth: number
  opacity: number
  brightness: number
  contrast: number
  highlight: boolean
  onSelect: (key: string | null) => void
  onNewBox: (geom: Geometrie) => void
  onChangeBox: (key: string, geom: Geometrie) => void
  onSupprGeste: (key: string) => void
}

const Canvas = (props: CanvasProps) => {
  const {
    boxes, classes, montrerRejetees, selectedKey, classeActive, image,
    imageSize, scale, strokeWidth, opacity, brightness, contrast, highlight,
  } = props

  const imageRef = useRef<Konva.Image | null>(null)

  useEffect(() => {
    const node = imageRef.current
    if (node && image) {
      node.cache()
      node.getLayer()?.batchDraw()
    }
  }, [image, brightness, contrast])

  const [dessin, setDessin] = useState<number[] | null>(null)
  const [souris, setSouris] = useState<{ x: number; y: number } | null>(null)
  const [zoom, setZoom] = useState(1.0)
  const [posStage, setPosStage] = useState({ x: 0, y: 0 })
  const [enPan, setEnPan] = useState(false)
  const panDepart = useRef<{ mx: number; my: number; sx: number; sy: number } | null>(null)
  // Geste 2 doigts (pinch-zoom + pan) : distance/centre/zoom/position initiaux
  const pinchDepart = useRef<{
    dist: number
    centre: { x: number; y: number }
    zoom: number
    sx: number
    sy: number
  } | null>(null)

  // Coordonnées viewport → contenu du Stage
  const versStage = (stage: Konva.Stage, vx: number, vy: number) => ({
    x: (vx - stage.x()) / stage.scaleX(),
    y: (vy - stage.y()) / stage.scaleY(),
  })

  // Positions des doigts relatives au conteneur du Stage
  const pointsTactiles = (stage: Konva.Stage, touches: TouchList) => {
    const rect = stage.container().getBoundingClientRect()
    return Array.from(touches).map((t) => ({
      x: t.clientX - rect.left,
      y: t.clientY - rect.top,
    }))
  }

  // Normalise une bbox : largeur/hauteur positives, bornée à l'image, min 5px
  const bornerRect = (r: Geometrie): Geometrie => {
    let { x, y, w, h } = r
    if (w < 0) {
      x += w
      w = -w
    }
    if (h < 0) {
      y += h
      h = -h
    }
    w += Math.min(0, x)
    x = Math.max(0, x)
    h += Math.min(0, y)
    y = Math.max(0, y)
    w = Math.max(5, Math.min(w, imageSize[0] - x))
    h = Math.max(5, Math.min(h, imageSize[1] - y))
    x = Math.min(Math.max(0, x), Math.max(0, imageSize[0] - w))
    y = Math.min(Math.max(0, y), Math.max(0, imageSize[1] - h))
    return { x, y, w, h }
  }

  // Nouvelle image chargée : zoom et position remis à zéro
  useEffect(() => {
    setZoom(1.0)
    setPosStage({ x: 0, y: 0 })
    setDessin(null)
  }, [imageSize[0], imageSize[1]])

  // Clic/tap sur zone vide : démarrer un dessin, ou désélectionner
  const clicZoneVide = (e: Konva.KonvaEventObject<MouseEvent | TouchEvent>) => {
    if (!(e.target instanceof Konva.Rect)) {
      const stage = e.target.getStage()!
      if (selectedKey === null) {
        const p = stage.getPointerPosition()!
        const sc = versStage(stage, p.x, p.y)
        setDessin([sc.x, sc.y, sc.x, sc.y])
      } else {
        props.onSelect(null)
      }
    }
  }

  // Fin de dessin (souris ou tactile) : crée la bbox si le geste est assez grand
  const finirDessin = () => {
    if (dessin === null) return
    const dw = Math.abs(dessin[2] - dessin[0])
    const dh = Math.abs(dessin[3] - dessin[1])
    if (dw * zoom < MIN_DESSIN_PX && dh * zoom < MIN_DESSIN_PX) {
      setDessin(null)
      return
    }
    props.onNewBox(
      bornerRect({
        x: dessin[0] / scale,
        y: dessin[1] / scale,
        w: (dessin[2] - dessin[0]) / scale,
        h: (dessin[3] - dessin[1]) / scale,
      }),
    )
    setDessin(null)
  }

  const zoomerSur = (
    pointeur: { x: number; y: number },
    ancien: number,
    nouveau: number,
    sx: number,
    sy: number,
  ) => {
    const origine = { x: (pointeur.x - sx) / ancien, y: (pointeur.y - sy) / ancien }
    setZoom(nouveau)
    setPosStage({
      x: pointeur.x - origine.x * nouveau,
      y: pointeur.y - origine.y * nouveau,
    })
  }

  const W = imageSize[0] * scale
  const H = imageSize[1] * scale

  const visibles = boxes.filter((b) => b.state !== "rejetee" || montrerRejetees)
  // La sélection est rendue en dernier (au-dessus), sans réordonner l'état
  const ordonnees = [
    ...visibles.filter((b) => b.key !== selectedKey),
    ...visibles.filter((b) => b.key === selectedKey),
  ]

  const couleurDessin = couleurClasse(classeActive)

  return (
    <Stage
      width={W}
      height={H}
      x={posStage.x}
      y={posStage.y}
      scaleX={zoom}
      scaleY={zoom}
      // touchAction none : le navigateur ne scrolle/zoome jamais la page
      // pendant qu'on manipule le canvas au doigt
      style={{ cursor: enPan ? "grabbing" : "crosshair", touchAction: "none" }}
      // ── Zoom molette, centré sur le curseur ────────────────────────────
      onWheel={(e) => {
        e.evt.preventDefault()
        const stage = e.target.getStage()!
        const ancien = stage.scaleX()
        const pointeur = stage.getPointerPosition()!
        const nouveau =
          e.evt.deltaY < 0
            ? Math.min(ancien * ZOOM_FACTEUR, ZOOM_MAX)
            : Math.max(ancien / ZOOM_FACTEUR, ZOOM_MIN)
        zoomerSur(pointeur, ancien, nouveau, stage.x(), stage.y())
      }}
      // ── Double-clic / double-tap : reset zoom ──────────────────────────
      onDblClick={() => {
        setZoom(1.0)
        setPosStage({ x: 0, y: 0 })
      }}
      onDblTap={() => {
        setZoom(1.0)
        setPosStage({ x: 0, y: 0 })
      }}
      // ── MouseDown : bouton droit = pan, gauche = dessin/désélection ────
      onMouseDown={(e) => {
        if (e.evt.button === 2) {
          e.evt.preventDefault()
          const stage = e.target.getStage()!
          const p = stage.getPointerPosition()!
          panDepart.current = { mx: p.x, my: p.y, sx: stage.x(), sy: stage.y() }
          setEnPan(true)
          return
        }
        if (e.evt.button === 0) clicZoneVide(e)
      }}
      onContextMenu={(e) => {
        e.evt.preventDefault()
      }}
      // ── MouseMove : pan, ou croix de visée + rect en cours ─────────────
      onMouseMove={(e) => {
        const stage = e.target.getStage()!
        const pointeur = stage.getPointerPosition()!
        if (panDepart.current) {
          setPosStage({
            x: panDepart.current.sx + (pointeur.x - panDepart.current.mx),
            y: panDepart.current.sy + (pointeur.y - panDepart.current.my),
          })
          return
        }
        const sc = versStage(stage, pointeur.x, pointeur.y)
        setSouris(sc)
        if (dessin !== null) setDessin([dessin[0], dessin[1], sc.x, sc.y])
      }}
      onMouseLeave={() => {
        setDessin(null)
        setSouris(null)
        panDepart.current = null
        setEnPan(false)
      }}
      // ── MouseUp : fin de pan ou création de bbox ───────────────────────
      onMouseUp={(e) => {
        if (e.evt.button === 2) {
          panDepart.current = null
          setEnPan(false)
          return
        }
        finirDessin()
      }}
      // ── TACTILE : 1 doigt = dessin/sélection · 2 doigts = zoom + pan ───
      onTouchStart={(e) => {
        e.evt.preventDefault()
        const stage = e.target.getStage()!
        const touches = e.evt.touches
        if (touches.length === 2) {
          // un 2e doigt posé annule le dessin en cours et démarre le pinch
          setDessin(null)
          const [p1, p2] = pointsTactiles(stage, touches)
          pinchDepart.current = {
            dist: Math.hypot(p2.x - p1.x, p2.y - p1.y),
            centre: { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 },
            zoom: stage.scaleX(),
            sx: stage.x(),
            sy: stage.y(),
          }
          return
        }
        if (touches.length === 1 && pinchDepart.current === null) clicZoneVide(e)
      }}
      onTouchMove={(e) => {
        e.evt.preventDefault()
        const stage = e.target.getStage()!
        const touches = e.evt.touches
        if (touches.length === 2 && pinchDepart.current) {
          const [p1, p2] = pointsTactiles(stage, touches)
          const depart = pinchDepart.current
          const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y)
          const centre = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 }
          const nouveau = Math.min(
            Math.max((depart.zoom * dist) / depart.dist, ZOOM_MIN),
            ZOOM_MAX,
          )
          // zoom autour du centre initial + pan si le centre se déplace
          const origine = {
            x: (depart.centre.x - depart.sx) / depart.zoom,
            y: (depart.centre.y - depart.sy) / depart.zoom,
          }
          setZoom(nouveau)
          setPosStage({
            x: centre.x - origine.x * nouveau,
            y: centre.y - origine.y * nouveau,
          })
          return
        }
        if (dessin !== null && touches.length === 1) {
          const pointeur = stage.getPointerPosition()
          if (pointeur) {
            const sc = versStage(stage, pointeur.x, pointeur.y)
            setDessin([dessin[0], dessin[1], sc.x, sc.y])
          }
        }
      }}
      onTouchEnd={(e) => {
        if (e.evt.touches.length < 2) pinchDepart.current = null
        if (e.evt.touches.length === 0) finirDessin()
      }}
    >
      {/* Couche image — le bitmap servi peut être une RÉDUCTION de
          l'original : on le remet à l'échelle du repère du fichier annoté
          (imageSize), dans lequel vivent boîtes et coordonnées */}
      <Layer>
        <KImage
          ref={imageRef}
          image={image}
          scaleX={image ? (imageSize[0] / image.naturalWidth) * scale : scale}
          scaleY={image ? (imageSize[1] / image.naturalHeight) * scale : scale}
          filters={[Konva.Filters.Brighten, Konva.Filters.Contrast]}
          brightness={brightness}
          contrast={contrast}
        />
      </Layer>

      {/* Couche bboxes + rect en cours de dessin */}
      <Layer>
        <Group opacity={opacity}>
          {ordonnees.map((b) => (
            <Box
              key={b.key}
              box={b}
              classes={classes}
              scale={scale}
              strokeWidth={strokeWidth}
              highlight={highlight}
              isSelected={b.key === selectedKey}
              onSelect={() => props.onSelect(b.key)}
              onSupprGeste={() => props.onSupprGeste(b.key)}
              onChange={(geom) => props.onChangeBox(b.key, bornerRect(geom))}
            />
          ))}
        </Group>
        {dessin !== null && (
          <Rect
            fill={couleurDessin + "4d"}
            stroke={couleurDessin}
            strokeWidth={1 / zoom}
            x={dessin[0]}
            y={dessin[1]}
            width={dessin[2] - dessin[0]}
            height={dessin[3] - dessin[1]}
          />
        )}
      </Layer>

      {/* Croix de visée — non interactive, toujours au-dessus */}
      {souris !== null && (
        <Layer listening={false}>
          <Line
            points={[souris.x, 0, souris.x, H]}
            stroke="rgba(0,0,0,0.45)"
            strokeWidth={3 / zoom}
            dash={[6 / zoom, 6 / zoom]}
          />
          <Line
            points={[souris.x, 0, souris.x, H]}
            stroke="rgba(255,255,255,0.9)"
            strokeWidth={1 / zoom}
            dash={[6 / zoom, 6 / zoom]}
          />
          <Line
            points={[0, souris.y, W, souris.y]}
            stroke="rgba(0,0,0,0.45)"
            strokeWidth={3 / zoom}
            dash={[6 / zoom, 6 / zoom]}
          />
          <Line
            points={[0, souris.y, W, souris.y]}
            stroke="rgba(255,255,255,0.9)"
            strokeWidth={1 / zoom}
            dash={[6 / zoom, 6 / zoom]}
          />
        </Layer>
      )}
    </Stage>
  )
}

export default Canvas
