/** Écran de campagne d'annotation.
 *
 * Règles d'ergonomie (fixées le 2026-08-03) :
 * - Entrée = enregistrer PUIS ouvrir la prochaine image à traiter ;
 *   Ctrl+S = enregistrer et rester (l'image est rouverte dans la foulée
 *   pour continuer sans dupliquer les boîtes) ;
 * - enregistrement IMPOSSIBLE tant qu'une proposition du modèle n'est pas
 *   tranchée (V valider, R/Suppr rejeter, Tab pour les parcourir) — le
 *   serveur le refuserait de toute façon (422) ;
 * - retoucher une proposition (déplacer, redimensionner, changer la classe)
 *   VAUT validation : on ne retouche pas une boîte sans l'avoir regardée ;
 * - brouillon local automatique par image (localStorage) : proposé à la
 *   réouverture, jamais imposé — le serveur reste la seule vérité ;
 * - annulation multi-niveaux (Ctrl+Z / Ctrl+Y), 100 niveaux ;
 * - à l'ouverture du lot, reprise là où il s'était arrêté : première image
 *   `en_cours`, sinon première `a_annoter` ;
 * - naviguer vers une image déjà annotée demande confirmation : l'ouvrir la
 *   repasse `en_cours` (réédition), ce n'est pas une consultation neutre ;
 * - une image quittée SANS enregistrement est refermée (le serveur la remet
 *   à son statut d'origine) : changer d'image, revenir aux lots ou fermer la
 *   page ne laisse pas d'image bloquée « en cours ». Le brouillon local
 *   survit et reste proposé à la réouverture ;
 * - relecture (annotateur confirmé ou administrateur, image annotée par
 *   quelqu'un d'autre) : corriger ou valider en l'état — l'image passe
 *   « relue ». Ponctuelle et facultative, jamais une étape obligatoire ;
 * - guide d'annotation intégré (G), filtre de la liste par statut et classe.
 */
import { useCallback, useEffect, useRef, useState } from "react"
import useImage from "use-image"

import { api, type ImageLot, type Moi, type Ouverture } from "./api"
import type { Geometrie } from "./Box"
import {
  cleClient,
  corpsEnregistrement,
  couleurClasse,
  depuisServeur,
  effacerBrouillon,
  lireBrouillon,
  propositionsRestantes,
  sauverBrouillon,
  type BoxItem,
} from "./boxes"
import Canvas from "./Canvas"
import Guide from "./Guide"

const HISTORIQUE_MAX = 100

const LIBELLES_STATUT: Record<string, string> = {
  en_attente_preannotation: "en file de pré-annotation",
  a_annoter: "à annoter",
  en_cours: "en cours",
  annotee: "annotée",
  relue: "relue",
}

const LIBELLES_COURTS: Record<string, string> = {
  en_attente_preannotation: "en file",
  a_annoter: "à annoter",
  en_cours: "en cours",
  annotee: "annotée",
  relue: "relue",
}

const formatChrono = (totalSecondes: number) => {
  const s = Math.max(0, Math.round(totalSecondes))
  return `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, "0")}`
}

interface Props {
  moi: Moi
  lotId: number
  lotNom: string
  onRetour: () => void
  surErreurAuth: (e: unknown) => boolean
}

export default function Annotation({ moi, lotId, lotNom, onRetour, surErreurAuth }: Props) {
  const [imagesLot, setImagesLot] = useState<ImageLot[] | null>(null)
  const [ouverte, setOuverte] = useState<Ouverture | null>(null)
  const [boxes, setBoxes] = useState<BoxItem[]>([])
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const [classeActive, setClasseActive] = useState(0)
  const [termine, setTermine] = useState(false)
  const [listeVisible, setListeVisible] = useState(false)
  const [guideVisible, setGuideVisible] = useState(false)
  // filtre de la liste : statut et classe présente ("tous" / -1 = sans filtre)
  const [filtreStatut, setFiltreStatut] = useState("tous")
  const [filtreClasse, setFiltreClasse] = useState(-1)
  // référentiel data.yaml, disponible même sans image ouverte (écran de fin)
  const [nomsClasses, setNomsClasses] = useState<string[]>([])

  const [erreur, setErreur] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)
  const [enEnregistrement, setEnEnregistrement] = useState(false)
  const [brouillonPropose, setBrouillonPropose] = useState<{
    quand: string
    boxes: BoxItem[]
  } | null>(null)

  // réglages d'affichage
  const [opacite, setOpacite] = useState(1.0)
  const [epaisseur, setEpaisseur] = useState(2)
  const [lumiere, setLumiere] = useState(0)
  const [contraste, setContraste] = useState(0)
  const [highlight, setHighlight] = useState(false)
  const [montrerRejetees, setMontrerRejetees] = useState(false)
  const [pleineResolution, setPleineResolution] = useState(false)

  // chrono par image
  const [ouvertA, setOuvertA] = useState<number>(Date.now())
  const [ecoule, setEcoule] = useState(0)
  useEffect(() => {
    const tick = () => setEcoule((Date.now() - ouvertA) / 1000)
    tick()
    const it = setInterval(tick, 1000)
    return () => clearInterval(it)
  }, [ouvertA])

  // ── Miroirs en ref pour le clavier et les callbacks stables ──────────────
  const boxesRef = useRef(boxes)
  boxesRef.current = boxes
  const selectedKeyRef = useRef(selectedKey)
  selectedKeyRef.current = selectedKey
  const imagesLotRef = useRef(imagesLot)
  imagesLotRef.current = imagesLot
  const ouverteRef = useRef(ouverte)
  ouverteRef.current = ouverte
  const statutCourantRef = useRef<string>("en_cours")
  const guideVisibleRef = useRef(false)
  guideVisibleRef.current = guideVisible
  // état des boîtes tel que le serveur le connaît (sérialisé) : sert au
  // brouillon (différence = travail non enregistré)
  const initialesRef = useRef("")

  useEffect(() => {
    api
      .classes()
      .then((r) => setNomsClasses(r.classes))
      .catch(() => {
        /* le filtre par classe restera vide — sans gravité */
      })
  }, [])

  // ── Historique (annulation multi-niveaux) ────────────────────────────────
  const passeRef = useRef<BoxItem[][]>([])
  const futurRef = useRef<BoxItem[][]>([])
  const [versionHistorique, setVersionHistorique] = useState(0)

  const commit = useCallback((suivantes: BoxItem[]) => {
    passeRef.current.push(boxesRef.current)
    if (passeRef.current.length > HISTORIQUE_MAX) passeRef.current.shift()
    futurRef.current = []
    setBoxes(suivantes)
    setVersionHistorique((v) => v + 1)
  }, [])

  const annulerHisto = useCallback(() => {
    const precedent = passeRef.current.pop()
    if (precedent === undefined) return
    futurRef.current.push(boxesRef.current)
    setBoxes(precedent)
    setSelectedKey(null)
    setVersionHistorique((v) => v + 1)
  }, [])

  const retablirHisto = useCallback(() => {
    const suivant = futurRef.current.pop()
    if (suivant === undefined) return
    passeRef.current.push(boxesRef.current)
    setBoxes(suivant)
    setSelectedKey(null)
    setVersionHistorique((v) => v + 1)
  }, [])

  // ── Opérations sur les boîtes ────────────────────────────────────────────

  const validerBoite = useCallback(
    (key: string) => {
      const boite = boxesRef.current.find((b) => b.key === key)
      if (!boite || boite.state === "validee") return
      commit(
        boxesRef.current.map((b) => (b.key === key ? { ...b, state: "validee" as const } : b)),
      )
    },
    [commit],
  )

  /** Geste « supprimer » : une boîte humaine disparaît, une boîte du modèle
   * est rejetée (jamais supprimée — ses faux positifs sont conservés). */
  const supprimerOuRejeter = useCallback(
    (key: string) => {
      const boite = boxesRef.current.find((b) => b.key === key)
      if (!boite || boite.state === "rejetee") return
      if (boite.source === "humain") {
        commit(boxesRef.current.filter((b) => b.key !== key))
      } else {
        commit(
          boxesRef.current.map((b) => (b.key === key ? { ...b, state: "rejetee" as const } : b)),
        )
      }
      setSelectedKey(null)
    },
    [commit],
  )

  const changerGeometrie = useCallback(
    (key: string, geom: Geometrie) => {
      // retouche = validation : on ne retouche pas sans avoir regardé
      commit(
        boxesRef.current.map((b) =>
          b.key === key
            ? { ...b, ...geom, state: b.state === "proposee" ? ("validee" as const) : b.state }
            : b,
        ),
      )
    },
    [commit],
  )

  const changerClasse = useCallback(
    (classeId: number) => {
      setClasseActive(classeId)
      const key = selectedKeyRef.current
      if (key === null) return
      const boite = boxesRef.current.find((b) => b.key === key)
      if (!boite || boite.state === "rejetee") return
      commit(
        boxesRef.current.map((b) =>
          b.key === key
            ? {
                ...b,
                classId: classeId,
                state: b.state === "proposee" ? ("validee" as const) : b.state,
              }
            : b,
        ),
      )
    },
    [commit],
  )

  const nouvelleBoite = useCallback(
    (geom: Geometrie) => {
      commit([
        ...boxesRef.current,
        {
          key: cleClient(),
          serverId: null,
          classId: classeActive,
          ...geom,
          source: "humain",
          state: "validee",
          confidence: null,
          origEtat: null,
        },
      ])
    },
    [commit, classeActive],
  )

  const cyclerPropositions = useCallback(() => {
    const propositions = boxesRef.current.filter((b) => b.state === "proposee")
    if (propositions.length === 0) return
    const idx = propositions.findIndex((b) => b.key === selectedKeyRef.current)
    setSelectedKey(propositions[(idx + 1) % propositions.length].key)
  }, [])

  // ── Ouverture / fermeture d'une image ────────────────────────────────────

  /** Referme l'image courante si elle est restée « en cours » sans
   * enregistrement : le serveur la remet à son statut d'origine (annotée si
   * elle l'était, à annoter sinon). Sans ce geste, changer d'image ou
   * quitter l'écran la laisserait bloquée « en cours » indéfiniment. Le
   * brouillon local n'est pas touché : il sera proposé à la réouverture.
   * `saufImageId` protège l'image qu'on est en train de (r)ouvrir. */
  const fermerCourante = useCallback((saufImageId?: number) => {
    const o = ouverteRef.current
    if (!o || o.id === saufImageId || statutCourantRef.current !== "en_cours") return
    api.fermerSansAttendre(o.id)
    // statut restitué, connu sans attendre la réponse : même règle que le
    // serveur (relue_par posé ⇔ relecture en vigueur ; annotated_at posé ⇔
    // elle était annotée)
    const statut = o.relue_par ? "relue" : o.deja_annotee ? "annotee" : "a_annoter"
    statutCourantRef.current = statut
    setImagesLot(
      (liste) => liste?.map((i) => (i.id === o.id ? { ...i, statut } : i)) ?? null,
    )
  }, [])

  // fermeture de la page (onglet fermé, rechargement, navigation externe) :
  // pagehide est le dernier moment fiable, sendBeacon part sans attendre
  useEffect(() => {
    const surSortie = () => fermerCourante()
    window.addEventListener("pagehide", surSortie)
    return () => window.removeEventListener("pagehide", surSortie)
  }, [fermerCourante])

  const ouvrirImage = useCallback(
    async (imageId: number, confirmerReouverture = false) => {
      const meta = imagesLotRef.current?.find((i) => i.id === imageId)
      if (
        confirmerReouverture &&
        meta &&
        (meta.statut === "annotee" || meta.statut === "relue") &&
        !window.confirm(
          `« ${meta.nom} » est déjà ${LIBELLES_STATUT[meta.statut]} — la rouvrir ` +
            "pour modification ? Elle repassera « en cours » jusqu'au prochain enregistrement.",
        )
      ) {
        return
      }
      fermerCourante(imageId)
      setErreur(null)
      setInfo(null)
      try {
        const o = await api.ouvrir(imageId)
        const initiales = o.boites.map((b) => depuisServeur(b, o.largeur, o.hauteur))
        setOuverte(o)
        statutCourantRef.current = "en_cours"
        setBoxes(initiales)
        initialesRef.current = JSON.stringify(initiales)
        setSelectedKey(null)
        passeRef.current = []
        futurRef.current = []
        setVersionHistorique((v) => v + 1)
        setTermine(false)
        setOuvertA(Date.now())
        setImagesLot(
          (liste) =>
            liste?.map((i) => (i.id === imageId ? { ...i, statut: "en_cours" } : i)) ?? null,
        )
        // brouillon local : proposé seulement s'il diffère de l'état serveur
        const brouillon = lireBrouillon(imageId)
        if (brouillon && JSON.stringify(brouillon.boxes) !== initialesRef.current) {
          setBrouillonPropose(brouillon)
        } else {
          if (brouillon) effacerBrouillon(imageId)
          setBrouillonPropose(null)
        }
      } catch (e) {
        if (!surErreurAuth(e)) setErreur(e instanceof Error ? e.message : String(e))
      }
    },
    [fermerCourante, surErreurAuth],
  )

  // Chargement du lot + reprise là où il s'était arrêté
  useEffect(() => {
    api
      .imagesDuLot(lotId)
      .then((liste) => {
        setImagesLot(liste)
        imagesLotRef.current = liste
        const cible =
          liste.find((i) => i.statut === "en_cours") ?? liste.find((i) => i.statut === "a_annoter")
        if (cible) {
          void ouvrirImage(cible.id)
        } else {
          // rien à annoter : le flux courant devient la réédition — la
          // liste des images est le point d'entrée, on l'ouvre d'office
          setTermine(true)
          setListeVisible(true)
        }
      })
      .catch((e) => {
        if (!surErreurAuth(e)) setErreur(e instanceof Error ? e.message : String(e))
      })
  }, [lotId, ouvrirImage, surErreurAuth])

  // ── Brouillon automatique (filet anti-perte, débordé de 400 ms) ─────────
  useEffect(() => {
    const o = ouverte
    if (!o) return
    const t = setTimeout(() => {
      if (JSON.stringify(boxes) !== initialesRef.current) sauverBrouillon(o.id, boxes)
      else effacerBrouillon(o.id)
    }, 400)
    return () => clearTimeout(t)
  }, [boxes, ouverte])

  // ── Navigation ───────────────────────────────────────────────────────────

  const indexCourant = useCallback(() => {
    const liste = imagesLotRef.current
    const o = ouverteRef.current
    if (!liste || !o) return -1
    return liste.findIndex((i) => i.id === o.id)
  }, [])

  /** Voisine dans l'ordre du lot, en sautant la file de pré-annotation
   * (ces images ne sont pas ouvrables). */
  const naviguer = useCallback(
    (delta: -1 | 1) => {
      const liste = imagesLotRef.current
      if (!liste) return
      let i = indexCourant()
      if (i < 0) return
      for (i += delta; i >= 0 && i < liste.length; i += delta) {
        if (liste[i].statut !== "en_attente_preannotation") {
          void ouvrirImage(liste[i].id, true)
          return
        }
      }
    },
    [indexCourant, ouvrirImage],
  )

  /** Prochaine image à traiter dans la campagne : après la courante puis en
   * rebouclant, la première `a_annoter` ou `en_cours`. */
  const prochaineCible = useCallback((depuisId: number): ImageLot | null => {
    const liste = imagesLotRef.current
    if (!liste || liste.length === 0) return null
    const depart = liste.findIndex((i) => i.id === depuisId)
    for (let pas = 1; pas <= liste.length; pas++) {
      const image = liste[(depart + pas) % liste.length]
      if (image.id !== depuisId && (image.statut === "a_annoter" || image.statut === "en_cours")) {
        return image
      }
    }
    return null
  }, [])

  // ── Enregistrement ───────────────────────────────────────────────────────

  const enregistrer = useCallback(
    async (avancer: boolean) => {
      const o = ouverteRef.current
      if (!o || enEnregistrement) return
      const restantes = propositionsRestantes(boxesRef.current)
      if (restantes > 0) {
        setErreur(
          `${restantes} proposition(s) du modèle restent à trancher — ` +
            "V pour valider, R pour rejeter, Tab pour les parcourir.",
        )
        return
      }
      setEnEnregistrement(true)
      setErreur(null)
      setInfo(null)
      try {
        // une image enregistrée entre-temps est repassée « annotée » :
        // la rouvrir d'abord (le PUT exige `en_cours`)
        if (statutCourantRef.current !== "en_cours") await api.ouvrir(o.id)
        await api.enregistrer(o.id, corpsEnregistrement(boxesRef.current, o.largeur, o.hauteur))
        statutCourantRef.current = "annotee"
        effacerBrouillon(o.id)
        setImagesLot(
          (liste) =>
            liste?.map((i) =>
              i.id === o.id ? { ...i, statut: "annotee", propositions: 0 } : i,
            ) ?? null,
        )
        if (avancer) {
          const cible = prochaineCible(o.id)
          if (cible) {
            void ouvrirImage(cible.id)
          } else {
            // plus rien À TRAITER : on est en relecture — Entrée suit
            // l'ordre de la liste quel que soit le statut, et l'écran de
            // fin n'apparaît qu'à la dernière image du lot
            const liste = imagesLotRef.current ?? []
            const idx = liste.findIndex((i) => i.id === o.id)
            let suivante: ImageLot | null = null
            for (let j = idx + 1; j < liste.length; j++) {
              if (liste[j].statut !== "en_attente_preannotation") {
                suivante = liste[j]
                break
              }
            }
            // geste délibéré (on vient d'enregistrer) : pas de confirmation
            if (suivante) void ouvrirImage(suivante.id)
            else setTermine(true)
          }
        } else {
          // rester : rouvrir pour continuer sur des identifiants serveur
          // frais (sans quoi un second enregistrement dupliquerait les
          // boîtes nouvelles)
          await ouvrirImage(o.id)
          setInfo("Enregistré — image rouverte pour continuer (Entrée pour clore et passer à la suivante).")
        }
      } catch (e) {
        if (!surErreurAuth(e)) setErreur(e instanceof Error ? e.message : String(e))
      } finally {
        setEnEnregistrement(false)
      }
    },
    [enEnregistrement, ouvrirImage, prochaineCible, surErreurAuth],
  )

  // ── Relecture ────────────────────────────────────────────────────────────
  // Valider (ou corriger) en relecture : l'image passe « relue » avec le
  // compte courant comme relecteur. Même exigence que l'enregistrement :
  // toutes les propositions tranchées. Puis on suit l'ordre de la liste,
  // comme Entrée en réédition.
  const relire = useCallback(async () => {
    const o = ouverteRef.current
    if (!o || enEnregistrement) return
    const restantes = propositionsRestantes(boxesRef.current)
    if (restantes > 0) {
      setErreur(
        `${restantes} proposition(s) du modèle restent à trancher — ` +
          "V pour valider, R pour rejeter, Tab pour les parcourir.",
      )
      return
    }
    setEnEnregistrement(true)
    setErreur(null)
    setInfo(null)
    try {
      if (statutCourantRef.current !== "en_cours") await api.ouvrir(o.id)
      await api.relire(o.id, corpsEnregistrement(boxesRef.current, o.largeur, o.hauteur))
      statutCourantRef.current = "relue"
      effacerBrouillon(o.id)
      setImagesLot(
        (liste) =>
          liste?.map((i) =>
            i.id === o.id ? { ...i, statut: "relue", propositions: 0 } : i,
          ) ?? null,
      )
      const liste = imagesLotRef.current ?? []
      const idx = liste.findIndex((i) => i.id === o.id)
      let suivante: ImageLot | null = null
      for (let j = idx + 1; j < liste.length; j++) {
        if (liste[j].statut !== "en_attente_preannotation") {
          suivante = liste[j]
          break
        }
      }
      if (suivante) void ouvrirImage(suivante.id)
      else setTermine(true)
    } catch (e) {
      if (!surErreurAuth(e)) setErreur(e instanceof Error ? e.message : String(e))
    } finally {
      setEnEnregistrement(false)
    }
  }, [enEnregistrement, ouvrirImage, surErreurAuth])

  // ── Clavier — un seul écouteur, actions lues via ref ─────────────────────
  const actionsRef = useRef({
    enregistrer,
    annulerHisto,
    retablirHisto,
    validerBoite,
    supprimerOuRejeter,
    changerClasse,
    cyclerPropositions,
    naviguer,
  })
  actionsRef.current = {
    enregistrer,
    annulerHisto,
    retablirHisto,
    validerBoite,
    supprimerOuRejeter,
    changerClasse,
    cyclerPropositions,
    naviguer,
  }
  const nbClassesRef = useRef(0)
  nbClassesRef.current = ouverte?.classes.length ?? 0

  useEffect(() => {
    const surTouche = (e: KeyboardEvent) => {
      const balise = (e.target as HTMLElement)?.tagName
      if (balise === "INPUT" || balise === "TEXTAREA" || balise === "SELECT") return
      const a = actionsRef.current

      // guide ouvert : Échap ou G le referment, tout le reste est neutralisé
      // (V, R, chiffres… agiraient sur des boîtes qu'on ne voit plus)
      if (guideVisibleRef.current) {
        if (e.key === "Escape" || e.key.toLowerCase() === "g") {
          e.preventDefault()
          setGuideVisible(false)
        }
        return
      }
      if (e.key.toLowerCase() === "g" && !e.ctrlKey && !e.metaKey) {
        setGuideVisible(true)
        return
      }

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
        e.preventDefault()
        void a.enregistrer(false)
        return
      }
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === "z") {
        e.preventDefault()
        a.annulerHisto()
        return
      }
      if (
        (e.ctrlKey || e.metaKey) &&
        (e.key.toLowerCase() === "y" || (e.shiftKey && e.key.toLowerCase() === "z"))
      ) {
        e.preventDefault()
        a.retablirHisto()
        return
      }
      if (e.ctrlKey || e.metaKey) return

      switch (e.key) {
        case "Enter":
          void a.enregistrer(true)
          return
        case "Tab":
          e.preventDefault()
          a.cyclerPropositions()
          return
        case "Escape":
          setSelectedKey(null)
          return
        case "Delete":
        case "Backspace":
          if (selectedKeyRef.current !== null) a.supprimerOuRejeter(selectedKeyRef.current)
          return
        case "ArrowLeft":
          a.naviguer(-1)
          return
        case "ArrowRight":
          a.naviguer(1)
          return
      }
      const touche = e.key.toLowerCase()
      if (touche === "v" && selectedKeyRef.current !== null) {
        a.validerBoite(selectedKeyRef.current)
        return
      }
      if (touche === "r" && selectedKeyRef.current !== null) {
        a.supprimerOuRejeter(selectedKeyRef.current)
        return
      }
      if (touche === "h") {
        setHighlight((prev) => !prev)
        return
      }
      const chiffre = parseInt(e.key, 10)
      if (!isNaN(chiffre) && chiffre >= 1 && chiffre <= nbClassesRef.current) {
        a.changerClasse(chiffre - 1)
      }
    }
    window.addEventListener("keydown", surTouche)
    return () => window.removeEventListener("keydown", surTouche)
  }, [])

  // La liste suit l'image courante (591 lignes : sans ça, on la cherche)
  useEffect(() => {
    if (!listeVisible) return
    document
      .querySelector(".ligne-image-courante")
      ?.scrollIntoView({ block: "nearest" })
  }, [listeVisible, ouverte])

  // ── Mise à l'échelle du canvas dans la zone disponible ───────────────────
  const zoneRef = useRef<HTMLDivElement>(null)
  const [dispo, setDispo] = useState({ w: 800, h: 600 })
  useEffect(() => {
    const el = zoneRef.current
    if (!el) return
    const mesurer = () =>
      setDispo({ w: Math.max(100, el.clientWidth - 16), h: Math.max(100, el.clientHeight - 16) })
    mesurer()
    const ro = new ResizeObserver(mesurer)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const [imageElement] = useImage(
    ouverte ? api.fichierUrl(ouverte.id, pleineResolution) : "",
  )

  // ── Dérivés d'affichage ──────────────────────────────────────────────────
  void versionHistorique // force la réévaluation des indicateurs ci-dessous
  const peutAnnuler = passeRef.current.length > 0
  const peutRetablir = futurRef.current.length > 0
  const restantes = propositionsRestantes(boxes)
  const rejetees = boxes.filter((b) => b.state === "rejetee").length
  const selection = selectedKey === null ? null : (boxes.find((b) => b.key === selectedKey) ?? null)
  const echelle = ouverte ? Math.min(dispo.w / ouverte.largeur, dispo.h / ouverte.hauteur, 1) : 1
  const position = (() => {
    if (!imagesLot || !ouverte) return null
    const i = imagesLot.findIndex((im) => im.id === ouverte.id)
    return i < 0 ? null : `${i + 1} / ${imagesLot.length}`
  })()
  const faites = imagesLot?.filter((i) => i.statut === "annotee" || i.statut === "relue").length ?? 0
  const enCoursN = imagesLot?.filter((i) => i.statut === "en_cours").length ?? 0
  const enFile =
    imagesLot?.filter((i) => i.statut === "en_attente_preannotation").length ?? 0
  // aucune boîte affichable (ni proposée ni validée) : l'écran doit dire si
  // c'est un négatif VALIDÉ ou une image jamais regardée
  const sansBoite =
    ouverte !== null && !boxes.some((b) => b.state !== "rejetee")
  // relecture possible : rôle confirmé/admin, image annotée par QUELQU'UN
  // D'AUTRE (le serveur re-vérifie tout, l'écran ne fait que proposer)
  const peutRelire =
    ouverte !== null &&
    !termine &&
    (moi.role === "annotateur_confirme" || moi.role === "administrateur") &&
    ouverte.annotee_par_id !== null &&
    ouverte.annotee_par_id !== moi.id
  // filtre de la liste (statut + classe présente) — la position « image
  // N / M » et la navigation ←/→ restent sur la liste complète
  const filtreActif = filtreStatut !== "tous" || filtreClasse >= 0
  const visibleDansListe = (img: ImageLot) =>
    (filtreStatut === "tous" || img.statut === filtreStatut) &&
    (filtreClasse < 0 || img.classes.includes(filtreClasse))
  const nbVisibles = imagesLot?.filter(visibleDansListe).length ?? 0

  // toute sortie vers l'écran des lots referme d'abord l'image courante
  const retournerAuxLots = () => {
    fermerCourante()
    onRetour()
  }

  const rendreLot = async () => {
    if (!window.confirm(`Marquer « ${lotNom} » terminé et rendre le verrou ?`)) return
    fermerCourante()
    try {
      await api.rendreLot(lotId, "termine")
      onRetour()
    } catch (e) {
      if (!surErreurAuth(e)) setErreur(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="page-annotation">
      <header className="barre">
        <button className="btn btn-petit" onClick={retournerAuxLots}>
          ← Lots
        </button>
        <button
          className={`btn btn-petit${listeVisible ? " btn-enfonce" : ""}`}
          onClick={() => setListeVisible((v) => !v)}
          title="Liste des images du lot : ouvrir n'importe laquelle, réédition comprise"
        >
          ☰ Images
        </button>
        <button
          className={`btn btn-petit${guideVisible ? " btn-enfonce" : ""}`}
          onClick={() => setGuideVisible((v) => !v)}
          title="Guide d'annotation : classes et règles (touche G)"
        >
          📖 Guide
        </button>
        <div className="barre-titre">
          <strong>{lotNom}</strong>
          {ouverte && !termine && (
            <>
              <span className="nom-image" title={ouverte.nom}>
                {ouverte.nom}
              </span>
              {/* libellé explicite : c'est une POSITION dans le lot, pas un
                  avancement — l'avancement (« X annotées / N ») est à droite */}
              {position && (
                <span
                  className="texte-doux"
                  title="Position dans le lot (ordre de la liste) — l'avancement est affiché à droite"
                >
                  image {position}
                </span>
              )}
              <span className="badge badge-chrono" title="Temps passé sur cette image">
                ⏱ {formatChrono(ecoule)}
              </span>
            </>
          )}
        </div>
        <div className="barre-droite">
          <span className="texte-doux">
            {faites} annotée{faites > 1 ? "s" : ""}
            {enCoursN > 0 ? ` · ${enCoursN} en cours` : ""} /{" "}
            {imagesLot?.length ?? "…"} · {moi.display_name ?? moi.username}
          </span>
        </div>
      </header>

      {erreur && <div className="bandeau bandeau-erreur">{erreur}</div>}
      {info && <div className="bandeau bandeau-info">{info}</div>}
      {brouillonPropose && (
        <div className="bandeau bandeau-brouillon">
          Brouillon local du {new Date(brouillonPropose.quand).toLocaleString("fr-FR")} trouvé
          pour cette image (travail non enregistré).
          <button
            className="btn btn-petit"
            onClick={() => {
              commit(brouillonPropose.boxes)
              setBrouillonPropose(null)
            }}
          >
            Reprendre le brouillon
          </button>
          <button
            className="btn btn-petit"
            onClick={() => {
              if (ouverte) effacerBrouillon(ouverte.id)
              setBrouillonPropose(null)
            }}
          >
            Ignorer
          </button>
        </div>
      )}
      {!termine && restantes > 0 && (
        <div className="bandeau bandeau-propositions">
          ⚠ {restantes} proposition{restantes > 1 ? "s" : ""} du modèle à trancher —{" "}
          <strong>V</strong> valider · <strong>R</strong> rejeter · <strong>Tab</strong>{" "}
          parcourir. L'enregistrement est bloqué tant qu'il en reste.
        </div>
      )}

      <div className="zone-centrale">
        {listeVisible && imagesLot && (
          <nav className="liste-images">
            <div className="filtres-liste">
              <select
                value={filtreStatut}
                onChange={(e) => setFiltreStatut(e.target.value)}
                title="Filtrer par statut"
              >
                <option value="tous">tous statuts</option>
                {Object.entries(LIBELLES_COURTS).map(([statut, libelle]) => (
                  <option key={statut} value={statut}>
                    {libelle}
                  </option>
                ))}
              </select>
              <select
                value={filtreClasse}
                onChange={(e) => setFiltreClasse(Number(e.target.value))}
                title="Filtrer par classe présente (boîtes proposées ou validées)"
              >
                <option value={-1}>toutes classes</option>
                {nomsClasses.map((nom, idx) => (
                  <option key={nom} value={idx}>
                    {nom}
                  </option>
                ))}
              </select>
              {filtreActif && (
                <div className="filtres-compte texte-doux">
                  {nbVisibles} / {imagesLot.length} image{nbVisibles > 1 ? "s" : ""}
                  <button
                    className="btn btn-petit"
                    onClick={() => {
                      setFiltreStatut("tous")
                      setFiltreClasse(-1)
                    }}
                  >
                    Tout afficher
                  </button>
                </div>
              )}
            </div>
            {imagesLot.map((img, idx) => {
              if (!visibleDansListe(img)) return null
              const courante = ouverte?.id === img.id && !termine
              const ouvrable = img.statut !== "en_attente_preannotation"
              return (
                <button
                  key={img.id}
                  className={`ligne-image${courante ? " ligne-image-courante" : ""}`}
                  disabled={!ouvrable}
                  title={
                    ouvrable
                      ? img.nom
                      : `${img.nom} — encore en file de pré-annotation : pas ouvrable`
                  }
                  onClick={() => {
                    if (!courante) void ouvrirImage(img.id, true)
                  }}
                >
                  <span className="ligne-image-num">{idx + 1}</span>
                  <span className="ligne-image-nom">{img.nom}</span>
                  {img.propositions > 0 && (
                    <span className="ligne-image-prop">{img.propositions} prop.</span>
                  )}
                  <span className={`badge badge-statut-${img.statut}`}>
                    {LIBELLES_COURTS[img.statut] ?? img.statut}
                  </span>
                </button>
              )
            })}
          </nav>
        )}
        <div className="zone-canvas" ref={zoneRef}>
          {termine ? (
            <div className="fin-de-lot">
              <h2>Plus rien à annoter dans ce lot 🎉</h2>
              <p>
                {faites} image{faites > 1 ? "s" : ""} annotée{faites > 1 ? "s" : ""}
                {enCoursN > 0 ? ` (et ${enCoursN} en cours)` : ""} sur{" "}
                {imagesLot?.length ?? 0}.
              </p>
              {enFile > 0 && (
                <p className="texte-doux">
                  {enFile} image{enFile > 1 ? "s" : ""} encore en file de pré-annotation — à
                  voir avec un administrateur (court-circuit possible depuis l'écran des lots).
                </p>
              )}
              <p className="texte-doux">
                Pour vérifier ou corriger une image, ouvrez-la depuis la liste
                ci-contre (☰ Images) — elle repassera « en cours » jusqu'au
                prochain enregistrement.
              </p>
              <div className="fin-de-lot-actions">
                {!listeVisible && (
                  <button className="btn" onClick={() => setListeVisible(true)}>
                    ☰ Parcourir les images
                  </button>
                )}
                <button className="btn btn-primaire" onClick={rendreLot}>
                  Marquer terminé et rendre le lot
                </button>
                <button className="btn" onClick={retournerAuxLots}>
                  Retour aux lots
                </button>
              </div>
            </div>
          ) : ouverte ? (
            <>
              {/* négatif validé ou image jamais regardée : sans cette
                  étiquette, les deux cas sont indiscernables à l'écran */}
              {sansBoite && (
                <div
                  className={`etiquette-vide ${
                    ouverte.deja_annotee ? "etiquette-negatif" : "etiquette-vierge"
                  }`}
                >
                  {ouverte.deja_annotee
                    ? "Aucun intrus — image validée (négatif)"
                    : "Aucune boîte tracée — image pas encore annotée"}
                </div>
              )}
              <Canvas
                boxes={boxes}
              classes={ouverte.classes}
              montrerRejetees={montrerRejetees}
              selectedKey={selectedKey}
              classeActive={classeActive}
              image={imageElement}
              imageSize={[ouverte.largeur, ouverte.hauteur]}
              scale={echelle}
              strokeWidth={epaisseur}
              opacity={opacite}
              brightness={lumiere}
              contrast={contraste}
              highlight={highlight}
              onSelect={(key) => {
                setSelectedKey(key)
                if (key !== null) {
                  const b = boxes.find((bx) => bx.key === key)
                  if (b && b.state !== "rejetee") setClasseActive(b.classId)
                }
              }}
              onNewBox={nouvelleBoite}
              onChangeBox={changerGeometrie}
              onSupprGeste={supprimerOuRejeter}
              />
            </>
          ) : (
            <div className="plein-ecran-message">{erreur ?? "Chargement de l'image…"}</div>
          )}
        </div>

        <aside className="panneau">
          {selection && (
            <div className="bloc-selection">
              <div className="bloc-selection-titre">
                Sélection : {ouverte?.classes[selection.classId] ?? selection.classId}
                {selection.source === "modele"
                  ? ` — modèle${selection.confidence != null ? ` · ${Math.round(selection.confidence * 100)} %` : ""}`
                  : " — humaine"}
              </div>
              <div className="bloc-selection-etat">
                {selection.state === "proposee" && <span className="badge badge-proposee">à trancher</span>}
                {selection.state === "validee" && <span className="badge badge-validee">validée</span>}
                {selection.state === "rejetee" && <span className="badge badge-rejetee">rejetée</span>}
              </div>
              <div className="rangee-boutons">
                {(selection.state === "proposee" || selection.state === "rejetee") && (
                  <button
                    className="btn btn-petit btn-primaire"
                    onClick={() => validerBoite(selection.key)}
                  >
                    ✓ {selection.state === "rejetee" ? "Restaurer" : "Valider"} (V)
                  </button>
                )}
                {selection.state !== "rejetee" && (
                  <button
                    className="btn btn-petit btn-danger"
                    onClick={() => supprimerOuRejeter(selection.key)}
                  >
                    ✗ {selection.source === "modele" ? "Rejeter" : "Supprimer"} (
                    {selection.source === "modele" ? "R" : "Suppr"})
                  </button>
                )}
              </div>
            </div>
          )}

          {ouverte && !termine && ouverte.annotee_par && (
            <div className="bloc-relecture">
              <div className="texte-doux">
                Annotée par <strong>{ouverte.annotee_par}</strong>
                {ouverte.annotee_le &&
                  ` le ${new Date(ouverte.annotee_le).toLocaleString("fr-FR")}`}
                {ouverte.relue_par && (
                  <>
                    <br />
                    Relue par <strong>{ouverte.relue_par}</strong>
                    {ouverte.relue_le &&
                      ` le ${new Date(ouverte.relue_le).toLocaleString("fr-FR")}`}
                  </>
                )}
              </div>
              {peutRelire && (
                <button
                  className="btn btn-large btn-relire"
                  disabled={enEnregistrement || restantes > 0}
                  title={
                    restantes > 0
                      ? "Impossible : des propositions du modèle restent à trancher"
                      : "Valider les boîtes telles qu'affichées (corrections comprises) : l'image passe « relue » à votre nom"
                  }
                  onClick={() => void relire()}
                >
                  {enEnregistrement ? "…" : "✓ Valider la relecture"}
                </button>
              )}
            </div>
          )}

          <div className="titre-panneau">Classe active</div>
          <div className="grille-classes">
            {(ouverte?.classes ?? []).map((nom, idx) => (
              <button
                key={nom}
                className={`btn-classe${idx === classeActive ? " btn-classe-active" : ""}`}
                style={{ borderColor: couleurClasse(idx) }}
                onClick={() => changerClasse(idx)}
                title={`Touche ${idx + 1}`}
              >
                <span className="pastille" style={{ background: couleurClasse(idx) }} />
                <span className="btn-classe-chiffre">{idx + 1}</span>
                <span className="btn-classe-nom">{nom}</span>
              </button>
            ))}
          </div>

          <button
            className="btn btn-primaire btn-large"
            disabled={!ouverte || termine || enEnregistrement || restantes > 0}
            title={
              restantes > 0
                ? "Impossible : des propositions du modèle restent à trancher"
                : "Entrée"
            }
            onClick={() => void enregistrer(true)}
          >
            {enEnregistrement ? "Enregistrement…" : "Enregistrer + suivante (Entrée)"}
          </button>
          <button
            className="btn btn-large"
            disabled={!ouverte || termine || enEnregistrement || restantes > 0}
            onClick={() => void enregistrer(false)}
          >
            Enregistrer (Ctrl+S)
          </button>

          <div className="rangee-boutons">
            <button className="btn btn-petit" disabled={!peutAnnuler} onClick={annulerHisto} title="Ctrl+Z">
              ↩ Annuler
            </button>
            <button className="btn btn-petit" disabled={!peutRetablir} onClick={retablirHisto} title="Ctrl+Y">
              ↪ Rétablir
            </button>
            <button
              className="btn btn-petit btn-danger"
              disabled={selection === null || selection.state === "rejetee"}
              onClick={() => selection && supprimerOuRejeter(selection.key)}
              title="Suppr"
            >
              🗑
            </button>
          </div>

          <div className="rangee-boutons">
            <button className="btn btn-petit" onClick={() => naviguer(-1)} title="←">
              ← Précédente
            </button>
            <button className="btn btn-petit" onClick={() => naviguer(1)} title="→">
              Suivante →
            </button>
          </div>

          <label className="ligne-case">
            <input
              type="checkbox"
              checked={montrerRejetees}
              onChange={(e) => setMontrerRejetees(e.target.checked)}
            />
            Afficher les rejetées ({rejetees})
          </label>
          <label className="ligne-case">
            <input type="checkbox" checked={highlight} onChange={(e) => setHighlight(e.target.checked)} />
            Surligner l'intérieur (H)
          </label>
          <label
            className="ligne-case"
            title="Par défaut le serveur envoie une version réduite (les coordonnées, normalisées, ne bougent pas). Cocher pour recharger l'original en pleine résolution."
          >
            <input
              type="checkbox"
              checked={pleineResolution}
              onChange={(e) => setPleineResolution(e.target.checked)}
            />
            Image originale pleine résolution
          </label>

          <div className="titre-panneau">Affichage</div>
          <label className="ligne-reglage">
            <span>Opacité</span>
            <input
              type="range"
              min={0.1}
              max={1}
              step={0.05}
              value={opacite}
              onChange={(e) => setOpacite(Number(e.target.value))}
            />
            <span className="valeur-reglage">{Math.round(opacite * 100)} %</span>
          </label>
          <label className="ligne-reglage">
            <span>Bordure</span>
            <input
              type="range"
              min={0.5}
              max={8}
              step={0.5}
              value={epaisseur}
              onChange={(e) => setEpaisseur(Number(e.target.value))}
            />
            <span className="valeur-reglage">{epaisseur}px</span>
          </label>
          <label className="ligne-reglage">
            <span>Lumière</span>
            <input
              type="range"
              min={-0.5}
              max={0.5}
              step={0.02}
              value={lumiere}
              onChange={(e) => setLumiere(Number(e.target.value))}
            />
            <span className="valeur-reglage">
              {lumiere >= 0 ? "+" : ""}
              {Math.round(lumiere * 100)}
            </span>
          </label>
          <label className="ligne-reglage">
            <span>Contraste</span>
            <input
              type="range"
              min={-50}
              max={50}
              step={2}
              value={contraste}
              onChange={(e) => setContraste(Number(e.target.value))}
            />
            <span className="valeur-reglage">
              {contraste >= 0 ? "+" : ""}
              {contraste}
            </span>
          </label>

          <div className="aide">
            <div className="titre-panneau">Raccourcis</div>
            <p>
              <strong>Entrée</strong> enregistrer + suivante · <strong>Ctrl+S</strong> enregistrer
              <br />
              <strong>V</strong> valider · <strong>R</strong>/<strong>Suppr</strong> rejeter ou
              supprimer · <strong>Tab</strong> proposition suivante
              <br />
              <strong>1–{ouverte?.classes.length ?? 8}</strong> classe · <strong>Échap</strong>{" "}
              désélectionner · <strong>H</strong> surlignage · <strong>G</strong> guide
              <br />
              <strong>Ctrl+Z</strong>/<strong>Ctrl+Y</strong> annuler / rétablir ·{" "}
              <strong>←</strong>/<strong>→</strong> naviguer
              <br />
              Clic <strong>molette</strong> : rejeter/supprimer · clic <strong>droit</strong> :
              déplacer la vue · <strong>molette</strong> : zoom
              <br />
              Tactile : <strong>1 doigt</strong> dessiner · <strong>2 doigts</strong>{" "}
              zoom/déplacer
            </p>
          </div>
        </aside>
      </div>

      {guideVisible && <Guide onFermer={() => setGuideVisible(false)} />}
    </div>
  )
}
