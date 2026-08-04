/** Client API — même origine que le serveur (build servi par FastAPI, ou
 * proxy Vite en dev) : le cookie de session HttpOnly fait tout, aucun jeton
 * côté JS. Toute erreur HTTP devient une ApiError portant le `detail`. */

export class ApiError extends Error {
  constructor(
    public status: number,
    detail: string,
  ) {
    super(detail)
  }
}

async function appel<T>(method: string, url: string, corps?: unknown): Promise<T> {
  const r = await fetch(url, {
    method,
    credentials: "same-origin",
    headers: corps !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: corps !== undefined ? JSON.stringify(corps) : undefined,
  })
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`
    try {
      const donnees = await r.json()
      if (typeof donnees.detail === "string") detail = donnees.detail
    } catch {
      /* corps non JSON : on garde le statut */
    }
    throw new ApiError(r.status, detail)
  }
  if (r.status === 204) return undefined as T
  return r.json()
}

// ── Formes des réponses (miroir des schémas Pydantic) ───────────────────────

export interface Moi {
  id: number
  username: string
  display_name: string | null
  role: string
  // vrai = mot de passe posé par un administrateur : l'écran de changement
  // s'impose avant tout le reste
  must_change_password: boolean
}

export interface Utilisateur {
  id: number
  username: string
  display_name: string | null
  role: string
  is_active: boolean
  must_change_password: boolean
  created_at: string
}

export interface Lot {
  id: number
  session_id: number
  session_name: string
  name: string
  holder: string | null
  holder_since: string | null
  total_images: number
  done_images: number
  in_progress_images: number
  parked_images: number
}

export interface ImageLot {
  id: number
  nom: string
  statut: string
  propositions: number
  // class_id présents (boîtes proposées ou validées) — trame du filtre
  classes: number[]
}

export interface BoiteServeur {
  id: number
  class_id: number
  x_center: number
  y_center: number
  box_width: number
  box_height: number
  source: "modele" | "humain"
  state: "proposee" | "validee" | "rejetee"
  confidence: number | null
}

export interface Ouverture {
  id: number
  batch_id: number
  session_id: number
  nom: string
  statut: string
  // annotée au moins une fois — survit à la réouverture, contrairement au
  // statut : c'est lui qui distingue « négatif validé » de « jamais regardée »
  deja_annotee: boolean
  // qui a annoté / relu, et quand — relue_par posé ⇔ relecture en vigueur
  annotee_par: string | null
  annotee_par_id: number | null
  annotee_le: string | null
  relue_par: string | null
  relue_le: string | null
  largeur: number
  hauteur: number
  classes: string[]
  boites: BoiteServeur[]
}

export interface BoiteEnvoi {
  id?: number
  class_id: number
  x_center: number
  y_center: number
  box_width: number
  box_height: number
  etat: "validee" | "rejetee"
}

export interface Enregistrement {
  statut: string
  validees: number
  rejetees: number
  supprimees: number
}

export interface Avancement {
  batch_id: number
  total: number
  par_statut: Record<string, number>
}

export interface SessionCapture {
  id: number
  name: string
  captured_on: string
  images: number
}

export interface RapportImport {
  session_name: string
  session_id: number | null
  batch_id: number | null
  created: string[]
  duplicates: [string, string][]
  rejected: [string, string][]
  renamed: [string, string][]
  aborted_reason: string | null
}

export interface RapportExport {
  output_dir: string
  images: number
  boxes: number
  empty_labels: number
  sessions: { id: number; name: string; images: number; boxes: number }[]
  class_counts: Record<string, number>
  renamed: [string, string][]
}

export interface ImageGaree {
  id: number
  nom: string
  lot: string
  session: string
  tentatives: number
  motif_type: string | null
  motif: string | null
}

export interface EtatFile {
  en_attente: number
  plafond_tentatives: number
  garees: ImageGaree[]
}

export interface ImportParams {
  source_dirs: string[]
  name: string
  rattacher: boolean
  captured_on?: string
  lighting?: string
  camera_height_cm?: number
  compost_state?: string
  operator?: string
  notes?: string
}

// ── Endpoints ───────────────────────────────────────────────────────────────

export const api = {
  login: (username: string, password: string) =>
    appel<Moi>("POST", "/api/auth/login", { username, password }),
  logout: () => appel<void>("POST", "/api/auth/logout"),
  moi: () => appel<Moi>("GET", "/api/auth/me"),
  changerMotDePasse: (actuel: string, nouveau: string) =>
    appel<Moi>("POST", "/api/auth/changer-mot-de-passe", { actuel, nouveau }),

  // gestion des comptes — administrateur uniquement
  utilisateurs: () => appel<Utilisateur[]>("GET", "/api/users"),
  creerUtilisateur: (corps: {
    username: string
    password: string
    display_name: string | null
    role: string
  }) => appel<Utilisateur>("POST", "/api/users", corps),
  modifierUtilisateur: (
    id: number,
    corps: Partial<{ is_active: boolean; role: string; display_name: string | null; password: string }>,
  ) => appel<Utilisateur>("PATCH", `/api/users/${id}`, corps),

  lots: () => appel<Lot[]>("GET", "/api/batches"),
  assignerLot: (lotId: number, userId: number) =>
    appel<{ batch_id: number }>("POST", `/api/batches/${lotId}/assign`, { user_id: userId }),
  decouperLot: (lotId: number, size: number) =>
    appel<{ created: { id: number; name: string; count: number }[]; moved: number }>(
      "POST",
      `/api/batches/${lotId}/split`,
      { size },
    ),
  avancement: (lotId: number) => appel<Avancement>("GET", `/api/batches/${lotId}/avancement`),
  imagesDuLot: (lotId: number) => appel<ImageLot[]>("GET", `/api/batches/${lotId}/images`),
  rendreLot: (lotId: number, reason: "rendu" | "termine") =>
    appel<{ batch_id: number }>("POST", `/api/batches/${lotId}/release`, { reason }),

  ouvrir: (imageId: number) => appel<Ouverture>("POST", `/api/images/${imageId}/ouvrir`),
  // par défaut le serveur sert une version réduite (coordonnées normalisées :
  // sans effet sur les boîtes) ; original=true force la pleine résolution
  fichierUrl: (imageId: number, original = false) =>
    `/api/images/${imageId}/fichier${original ? "?original=true" : ""}`,
  enregistrer: (imageId: number, boites: BoiteEnvoi[]) =>
    appel<Enregistrement>("PUT", `/api/images/${imageId}/annotations`, { boites }),
  // relecture (annotateur confirmé ou admin, image annotée par quelqu'un
  // d'autre) : corriger ou valider en l'état — l'image passe « relue »
  relire: (imageId: number, boites: BoiteEnvoi[]) =>
    appel<Enregistrement>("POST", `/api/images/${imageId}/relire`, { boites }),
  // référentiel data.yaml — pour les écrans sans image ouverte (filtre)
  classes: () => appel<{ classes: string[] }>("GET", "/api/images/classes"),
  // referme une image quittée sans enregistrement (retour à son statut
  // d'origine côté serveur). sendBeacon : la requête part sans attendre la
  // réponse et survit à la fermeture de la page — le même geste sert pour la
  // navigation interne, où un échec est sans gravité (l'image reste
  // simplement « en cours », comme avant l'existence de ce geste)
  fermerSansAttendre: (imageId: number) => {
    navigator.sendBeacon(`/api/images/${imageId}/fermer`)
  },
  passerAAnnoter: (imageIds: number[]) =>
    appel<{ passees: number }>("POST", "/api/images/passer-a-annoter", { image_ids: imageIds }),

  // ── Administration technique (administrateur) ─────────────────────────────
  sessions: () => appel<SessionCapture[]>("GET", "/api/sessions"),
  exporter: (outputDir: string, sessionNames: string[] | null) =>
    appel<RapportExport>("POST", "/api/exports", {
      output_dir: outputDir,
      session_names: sessionNames,
    }),
  etatFile: () => appel<EtatFile>("GET", "/api/preannotation/etat"),
  relancerGarees: (imageIds: number[]) =>
    appel<{ relancees: number }>("POST", "/api/preannotation/relancer", { image_ids: imageIds }),
  // l'import a un cas à part : un refus « rien d'importable » (409) porte le
  // rapport COMPLET en détail — on le restitue au lieu de le réduire à un
  // message d'erreur
  importer: async (corps: ImportParams): Promise<RapportImport> => {
    const r = await fetch("/api/imports", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(corps),
    })
    const donnees = await r.json().catch(() => null)
    if (r.ok) return donnees as RapportImport
    if (r.status === 409 && donnees && typeof donnees.detail === "object") {
      return donnees.detail as RapportImport
    }
    throw new ApiError(
      r.status,
      donnees && typeof donnees.detail === "string"
        ? donnees.detail
        : `${r.status} ${r.statusText}`,
    )
  },
}
