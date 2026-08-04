/** Liste des lots : les miens d'abord (annotation), tous ensuite. Pour un
 * administrateur : découpage en lots de N (round-robin entre postes côté
 * serveur), assignation à un compte actif, libération forcée, court-circuit
 * de la file de pré-annotation — plus les entrées vers les écrans Comptes et
 * Technique. */
import { useCallback, useEffect, useState } from "react"

import { api, type Lot, type Moi, type Utilisateur } from "./api"

interface Props {
  moi: Moi
  onOuvrirLot: (lotId: number, lotNom: string) => void
  onOuvrirComptes: () => void
  onOuvrirTechnique: () => void
  onDeconnexion: () => void
  surErreurAuth: (e: unknown) => boolean
}

const formatDepuis = (iso: string) => {
  const quand = new Date(iso)
  const jours = Math.floor((Date.now() - quand.getTime()) / 86_400_000)
  const date = quand.toLocaleDateString("fr-FR")
  if (jours <= 0) return `depuis aujourd'hui (${date})`
  return `depuis ${jours} jour${jours > 1 ? "s" : ""} (${date})`
}

export default function Lots({
  moi,
  onOuvrirLot,
  onOuvrirComptes,
  onOuvrirTechnique,
  onDeconnexion,
  surErreurAuth,
}: Props) {
  const [lots, setLots] = useState<Lot[] | null>(null)
  const [comptes, setComptes] = useState<Utilisateur[]>([])
  const [assignation, setAssignation] = useState<Record<number, string>>({})
  const [message, setMessage] = useState<string | null>(null)
  const [erreur, setErreur] = useState<string | null>(null)
  const admin = moi.role === "administrateur"

  const charger = useCallback(() => {
    api
      .lots()
      .then(setLots)
      .catch((e) => {
        if (!surErreurAuth(e)) setErreur(e instanceof Error ? e.message : String(e))
      })
  }, [surErreurAuth])

  useEffect(charger, [charger])

  // l'assignation a besoin de la liste des comptes actifs (admin seulement)
  useEffect(() => {
    if (!admin) return
    api
      .utilisateurs()
      .then((tous) => setComptes(tous.filter((u) => u.is_active)))
      .catch((e) => {
        if (!surErreurAuth(e)) setErreur(e instanceof Error ? e.message : String(e))
      })
  }, [admin, surErreurAuth])

  const agir = async (action: () => Promise<unknown>, succes: string) => {
    setMessage(null)
    setErreur(null)
    try {
      await action()
      setMessage(succes)
      charger()
    } catch (e) {
      if (!surErreurAuth(e)) setErreur(e instanceof Error ? e.message : String(e))
    }
  }

  // Geste admin : passer toutes les images encore en file de ce lot à
  // « à annoter », sans pré-annotation. Tracé côté serveur avec l'admin
  // comme auteur — jamais confondu avec un passage du worker.
  const courtCircuiter = async (lot: Lot) => {
    setMessage(null)
    setErreur(null)
    try {
      const images = await api.imagesDuLot(lot.id)
      const enAttente = images
        .filter((i) => i.statut === "en_attente_preannotation")
        .map((i) => i.id)
      if (enAttente.length === 0) {
        setMessage(`« ${lot.name} » : aucune image en file de pré-annotation.`)
        return
      }
      if (
        !window.confirm(
          `Passer ${enAttente.length} image(s) de « ${lot.name} » à « à annoter » ` +
            "sans pré-annotation ? Le worker ne les traitera plus.",
        )
      ) {
        return
      }
      const r = await api.passerAAnnoter(enAttente)
      setMessage(`« ${lot.name} » : ${r.passees} image(s) passée(s) à « à annoter ».`)
      charger()
    } catch (e) {
      if (!surErreurAuth(e)) setErreur(e instanceof Error ? e.message : String(e))
    }
  }

  const decouper = async (lot: Lot) => {
    const reponse = window.prompt(
      `Découper « ${lot.name} » (session ${lot.session_name}) en lots de N images.\n` +
        "Seules les images non entamées bougent ; le découpage alterne les postes de capture.\nN :",
      "50",
    )
    if (reponse === null) return
    const taille = parseInt(reponse, 10)
    if (isNaN(taille) || taille < 1) {
      setErreur("Taille de lot invalide.")
      return
    }
    setMessage(null)
    setErreur(null)
    try {
      const r = await api.decouperLot(lot.id, taille)
      setMessage(
        `« ${lot.name} » : ${r.moved} image(s) réparties en ${r.created.length} lot(s) — ` +
          r.created.map((c) => `${c.name} (${c.count})`).join(", ") +
          ".",
      )
      charger()
    } catch (e) {
      if (!surErreurAuth(e)) setErreur(e instanceof Error ? e.message : String(e))
    }
  }

  const assigner = (lot: Lot) => {
    const choix = assignation[lot.id]
    if (!choix) return
    const compte = comptes.find((u) => String(u.id) === choix)
    void agir(
      () => api.assignerLot(lot.id, Number(choix)),
      `« ${lot.name} » assigné à ${compte?.display_name ?? compte?.username ?? choix}.`,
    )
  }

  const libererForce = (lot: Lot) => {
    if (
      !window.confirm(
        `Libérer de force « ${lot.name} », détenu par ${lot.holder} ? ` +
          "Ses images « en cours » retourneront à leur statut d'origine.",
      )
    ) {
      return
    }
    void agir(
      () => api.rendreLot(lot.id, "rendu"),
      `« ${lot.name} » libéré (${lot.holder} n'en est plus détenteur).`,
    )
  }

  if (lots === null) {
    return <div className="plein-ecran-message">{erreur ?? "Chargement des lots…"}</div>
  }

  const miens = lots.filter((l) => l.holder === moi.username)
  const autres = lots.filter((l) => l.holder !== moi.username)

  const avancement = (lot: Lot) => {
    const restantes =
      lot.total_images - lot.done_images - lot.in_progress_images - lot.parked_images
    return (
      <div className="avancement-lot">
        <strong>
          {lot.done_images} / {lot.total_images}
        </strong>
        <span className="texte-doux">
          {lot.done_images} annotée{lot.done_images > 1 ? "s" : ""}
          {lot.in_progress_images > 0 && ` · ${lot.in_progress_images} en cours`}
          {restantes > 0 && ` · ${restantes} restante${restantes > 1 ? "s" : ""}`}
          {lot.parked_images > 0 && ` · ${lot.parked_images} garée${lot.parked_images > 1 ? "s" : ""}`}
        </span>
      </div>
    )
  }

  const ligne = (lot: Lot, actionnable: boolean) => (
    <tr key={lot.id}>
      <td>{lot.session_name}</td>
      <td>{lot.name}</td>
      <td className="cellule-nombre">{avancement(lot)}</td>
      <td>
        {lot.holder ?? "—"}
        {lot.holder && lot.holder_since && (
          <div className="texte-doux">{formatDepuis(lot.holder_since)}</div>
        )}
      </td>
      <td className="cellule-actions">
        {actionnable && (
          <button className="btn btn-primaire btn-petit" onClick={() => onOuvrirLot(lot.id, lot.name)}>
            Annoter
          </button>
        )}
        {admin && !actionnable && (
          <button className="btn btn-petit" onClick={() => onOuvrirLot(lot.id, lot.name)}>
            Ouvrir
          </button>
        )}
        {admin && (
          <>
            {lot.holder === null ? (
              <span className="groupe-assignation">
                <select
                  value={assignation[lot.id] ?? ""}
                  onChange={(e) =>
                    setAssignation((avant) => ({ ...avant, [lot.id]: e.target.value }))
                  }
                >
                  <option value="">— compte —</option>
                  {comptes.map((u) => (
                    <option key={u.id} value={u.id}>
                      {u.display_name ?? u.username}
                    </option>
                  ))}
                </select>
                <button
                  className="btn btn-petit"
                  disabled={!assignation[lot.id]}
                  onClick={() => assigner(lot)}
                >
                  Assigner
                </button>
              </span>
            ) : (
              <button
                className="btn btn-petit btn-danger"
                title="Libérer le verrou sans attendre le détenteur — ses images « en cours » retournent à leur statut d'origine"
                onClick={() => libererForce(lot)}
              >
                Libérer
              </button>
            )}
            <button
              className="btn btn-petit"
              title="Découper en lots de N images (round-robin entre postes de capture) — refusé si le lot est verrouillé"
              disabled={lot.holder !== null}
              onClick={() => decouper(lot)}
            >
              Découper…
            </button>
            <button
              className="btn btn-petit"
              title="Passer les images encore en file de pré-annotation directement à « à annoter » (worker indisponible, images garées, lot à ne pas pré-annoter)"
              onClick={() => courtCircuiter(lot)}
            >
              Court-circuiter la file
            </button>
          </>
        )}
      </td>
    </tr>
  )

  const table = (contenu: Lot[], actionnable: boolean) => (
    <table className="table-lots">
      <thead>
        <tr>
          <th>Session</th>
          <th>Lot</th>
          <th>Avancement</th>
          <th>Détenteur</th>
          <th></th>
        </tr>
      </thead>
      <tbody>{contenu.map((l) => ligne(l, actionnable))}</tbody>
    </table>
  )

  return (
    <div className="page">
      <header className="barre">
        <h1>Lots</h1>
        <div className="barre-droite">
          {admin && (
            <>
              <button className="btn btn-petit" onClick={onOuvrirComptes}>
                Comptes
              </button>
              <button className="btn btn-petit" onClick={onOuvrirTechnique}>
                Technique
              </button>
            </>
          )}
          <span className="texte-doux">
            {moi.display_name ?? moi.username} · {moi.role}
          </span>
          <button className="btn btn-petit" onClick={onDeconnexion}>
            Déconnexion
          </button>
        </div>
      </header>

      {message && <div className="bandeau bandeau-info">{message}</div>}
      {erreur && <div className="bandeau bandeau-erreur">{erreur}</div>}

      <h2>Mes lots</h2>
      {miens.length > 0 ? table(miens, true) : <p className="texte-doux">Aucun lot ne vous est assigné.</p>}

      {autres.length > 0 && (
        <>
          <h2>Tous les lots</h2>
          {table(autres, false)}
        </>
      )}
    </div>
  )
}
