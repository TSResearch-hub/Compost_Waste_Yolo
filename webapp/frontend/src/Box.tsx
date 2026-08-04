/** Une bounding box sur le canvas. La distinction d'état est LE point
 * critique : si une proposition du modèle ressemble à une boîte validée,
 * l'annotateur valide sans regarder et la pré-annotation dégrade le dataset.
 *
 * - proposée (modèle, à trancher) : trait ORANGE en tirets, intérieur hachuré
 *   à la couleur de classe, pastille orange « ? Classe · NN % » ;
 * - validée : trait PLEIN couleur de classe, pastille pleine (préfixe ✓ si
 *   elle vient du modèle) ;
 * - rejetée (visible sur demande) : grise, tirets, estompée, ni déplaçable
 *   ni redimensionnable — cliquable pour la restaurer.
 */
import Konva from "konva"
import React, { useEffect, useRef, useState } from "react"
import { Label, Rect, Tag, Text, Transformer } from "react-konva"

import { couleurClasse, GRIS_REJET, ORANGE_PROPOSITION, type BoxItem } from "./boxes"

export interface Geometrie {
  x: number
  y: number
  w: number
  h: number
}

interface BoxProps {
  box: BoxItem
  classes: string[]
  scale: number
  isSelected: boolean
  strokeWidth: number
  highlight: boolean
  onSelect: () => void
  onSupprGeste: () => void
  onChange: (geom: Geometrie) => void
}

const hexVersRgba = (hex: string, alpha: number): string => {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `rgba(${r},${g},${b},${alpha})`
}

// Motif de hachures diagonales par couleur de classe (mis en cache) : la
// classe reste lisible dans la boîte pendant que l'orange dit « à trancher »
const _hachures = new Map<string, HTMLCanvasElement>()
function motifHachures(couleur: string): HTMLCanvasElement {
  let c = _hachures.get(couleur)
  if (!c) {
    c = document.createElement("canvas")
    c.width = c.height = 12
    const ctx = c.getContext("2d")!
    ctx.strokeStyle = couleur
    ctx.globalAlpha = 0.4
    ctx.lineWidth = 2.5
    ctx.beginPath()
    // diagonale principale + répliques de coin pour un motif raccordable
    ctx.moveTo(-3, 15)
    ctx.lineTo(15, -3)
    ctx.moveTo(-3, 3)
    ctx.lineTo(3, -3)
    ctx.moveTo(9, 15)
    ctx.lineTo(15, 9)
    ctx.stroke()
    _hachures.set(couleur, c)
  }
  return c
}

const Box = (props: BoxProps) => {
  const { box, classes, scale, isSelected, strokeWidth, highlight } = props
  const rectRef = useRef<Konva.Rect>(null)
  const trRef = useRef<Konva.Transformer>(null)
  const [enMouvement, setEnMouvement] = useState(false)

  useEffect(() => {
    if (isSelected && trRef.current && rectRef.current) {
      trRef.current.nodes([rectRef.current])
      trRef.current.getLayer()?.batchDraw()
    }
  }, [isSelected])

  const couleur = couleurClasse(box.classId)
  const proposee = box.state === "proposee"
  const rejetee = box.state === "rejetee"
  const trait = rejetee ? GRIS_REJET : proposee ? ORANGE_PROPOSITION : couleur

  const remplissage: Partial<Konva.RectConfig> = proposee
    ? {
        fillPatternImage: motifHachures(couleur) as unknown as HTMLImageElement,
        fillPatternRepeat: "repeat",
      }
    : rejetee
      ? { fill: "rgba(120,120,120,0.12)" }
      : highlight
        ? { fill: hexVersRgba(couleur, 0.28) }
        : {}

  const nomClasse = classes[box.classId] ?? `classe ${box.classId}`
  const texte = proposee
    ? `? ${nomClasse}${box.confidence != null ? ` · ${Math.round(box.confidence * 100)} %` : ""}`
    : rejetee
      ? `${nomClasse} · rejetée`
      : box.source === "modele"
        ? `✓ ${nomClasse}`
        : nomClasse

  return (
    <React.Fragment>
      <Rect
        ref={rectRef}
        x={box.x * scale}
        y={box.y * scale}
        width={box.w * scale}
        height={box.h * scale}
        stroke={trait}
        strokeWidth={isSelected ? strokeWidth + 1 : strokeWidth}
        dash={proposee || rejetee ? [8, 4] : undefined}
        opacity={rejetee ? 0.55 : 1}
        {...remplissage}
        draggable={isSelected && !rejetee}
        onClick={props.onSelect}
        onTap={props.onSelect}
        onMouseDown={(e) => {
          // clic molette : rejeter (modèle) ou supprimer (humaine)
          if (e.evt.button === 1) {
            e.evt.preventDefault()
            props.onSupprGeste()
          }
        }}
        onDragStart={() => setEnMouvement(true)}
        onDragEnd={(e) => {
          setEnMouvement(false)
          props.onChange({
            x: e.target.x() / scale,
            y: e.target.y() / scale,
            w: box.w,
            h: box.h,
          })
        }}
        onTransformStart={() => setEnMouvement(true)}
        onTransformEnd={() => {
          const node = rectRef.current!
          const sx = node.scaleX()
          const sy = node.scaleY()
          node.scaleX(1)
          node.scaleY(1)
          setEnMouvement(false)
          props.onChange({
            x: node.x() / scale,
            y: node.y() / scale,
            w: Math.max(5, (node.width() * sx) / scale),
            h: Math.max(5, (node.height() * sy) / scale),
          })
        }}
      />
      {/* pastille d'état — masquée pendant un déplacement/redimensionnement */}
      {!enMouvement && (
        <Label x={box.x * scale + 2} y={box.y * scale + 2} listening={false}>
          <Tag fill={trait} opacity={0.92} cornerRadius={2} />
          <Text
            text={texte}
            fontSize={13}
            fontStyle="bold"
            fill={proposee ? "#1a1a1a" : "#ffffff"}
            padding={3}
          />
        </Label>
      )}
      {isSelected && !rejetee && (
        <Transformer
          ref={trRef}
          rotateEnabled={false}
          keepRatio={false}
          anchorSize={12}
          anchorCornerRadius={3}
          borderStroke={trait}
          borderStrokeWidth={strokeWidth}
          boundBoxFunc={(oldBox, newBox) =>
            newBox.width < 5 || newBox.height < 5 ? oldBox : newBox
          }
        />
      )}
    </React.Fragment>
  )
}

export default Box
