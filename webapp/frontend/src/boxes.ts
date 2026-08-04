/** Modèle client des boîtes et conversions YOLO ↔ pixels.
 *
 * Règle centrale, alignée sur l'API : une boîte du modèle traverse trois
 * états — `proposee` (à trancher), `validee`, `rejetee` — et n'est jamais
 * supprimée ; une boîte humaine est `validee` ou n'existe pas.
 * L'enregistrement est impossible tant qu'une `proposee` subsiste. */
import type { BoiteEnvoi, BoiteServeur } from "./api"

export type Etat = "proposee" | "validee" | "rejetee"

export interface BoxItem {
  key: string // clé client stable (id serveur ou clé générée)
  serverId: number | null // null = nouvelle boîte humaine
  classId: number
  // géométrie en pixels du FICHIER ANNOTÉ (crop si présent)
  x: number
  y: number
  w: number
  h: number
  source: "modele" | "humain"
  state: Etat
  confidence: number | null
  // état côté serveur au chargement : une boîte déjà rejetée en base et non
  // touchée est OMISE de l'envoi (ne pas ré-estampiller le décideur)
  origEtat: Etat | null
}

// Palette de classes — index = class_id du data.yaml. L'ORANGE est exclu :
// c'est la couleur réservée au signal « proposition du modèle ».
const PALETTE = [
  "#e6194b", // Plastique
  "#4363d8", // Metal
  "#9a6324", // Carton
  "#42d4f4", // Aluminium
  "#f032e6", // Ceramique
  "#3cb44b", // Verre
  "#911eb4", // Composite
  "#ffe119", // Eponge
]

export const ORANGE_PROPOSITION = "#ff8800"
export const GRIS_REJET = "#8a8a8a"

export function couleurClasse(classId: number): string {
  return PALETTE[classId % PALETTE.length]
}

let compteur = 0
export function cleClient(): string {
  return `neuve-${Date.now()}-${compteur++}`
}

export function depuisServeur(b: BoiteServeur, largeur: number, hauteur: number): BoxItem {
  return {
    key: `srv-${b.id}`,
    serverId: b.id,
    classId: b.class_id,
    x: (b.x_center - b.box_width / 2) * largeur,
    y: (b.y_center - b.box_height / 2) * hauteur,
    w: b.box_width * largeur,
    h: b.box_height * hauteur,
    source: b.source,
    state: b.state,
    confidence: b.confidence,
    origEtat: b.state,
  }
}

const borne = (v: number, min: number, max: number) => Math.min(Math.max(v, min), max)

function versYolo(b: BoxItem, largeur: number, hauteur: number) {
  const w = borne(b.w / largeur, 1e-6, 1)
  const h = borne(b.h / hauteur, 1e-6, 1)
  return {
    class_id: b.classId,
    x_center: borne(b.x / largeur + w / 2, 0, 1),
    y_center: borne(b.y / hauteur + h / 2, 0, 1),
    box_width: w,
    box_height: h,
  }
}

export function propositionsRestantes(boxes: BoxItem[]): number {
  return boxes.filter((b) => b.state === "proposee").length
}

/** Construit le corps d'enregistrement (état complet). Lève si une
 * proposition n'est pas tranchée — le serveur refuserait de toute façon. */
export function corpsEnregistrement(
  boxes: BoxItem[],
  largeur: number,
  hauteur: number,
): BoiteEnvoi[] {
  const corps: BoiteEnvoi[] = []
  for (const b of boxes) {
    if (b.state === "proposee") {
      throw new Error("Des propositions du modèle restent à trancher")
    }
    if (b.state === "rejetee") {
      // déjà rejetée en base et non restaurée : omise, elle le reste telle
      // quelle sans ré-estampiller le décideur
      if (b.origEtat === "rejetee") continue
      corps.push({ id: b.serverId!, ...versYolo(b, largeur, hauteur), etat: "rejetee" })
      continue
    }
    corps.push({
      ...(b.serverId !== null ? { id: b.serverId } : {}),
      ...versYolo(b, largeur, hauteur),
      etat: "validee",
    })
  }
  return corps
}

// ── Brouillon local (sauvegarde automatique, par image) ─────────────────────
// Filet anti-perte : navigation sans enregistrer, onglet fermé, panne. Le
// serveur reste la seule vérité — le brouillon est proposé, jamais imposé.

interface Brouillon {
  quand: string
  boxes: BoxItem[]
}

const cleBrouillon = (imageId: number) => `compost_brouillon_${imageId}`

export function sauverBrouillon(imageId: number, boxes: BoxItem[]): void {
  try {
    localStorage.setItem(
      cleBrouillon(imageId),
      JSON.stringify({ quand: new Date().toISOString(), boxes }),
    )
  } catch {
    /* stockage plein ou indisponible : le brouillon est un filet, pas un dû */
  }
}

export function lireBrouillon(imageId: number): Brouillon | null {
  try {
    const brut = localStorage.getItem(cleBrouillon(imageId))
    return brut ? (JSON.parse(brut) as Brouillon) : null
  } catch {
    return null
  }
}

export function effacerBrouillon(imageId: number): void {
  try {
    localStorage.removeItem(cleBrouillon(imageId))
  } catch {
    /* idem */
  }
}
