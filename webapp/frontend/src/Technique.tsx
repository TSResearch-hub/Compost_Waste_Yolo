/** Administration technique — import de sessions, export YOLO, file de
 * pré-annotation. Remplace les commandes CLI pour la passation : les chemins
 * restent des chemins CÔTÉ SERVEUR (l'écran pilote, il ne téléverse pas).
 * Le worker n'est pas piloté d'ici : il tourne à part, l'écran observe sa
 * file et peut y remettre une image garée. */
import { type FormEvent, useCallback, useEffect, useState } from "react"

import {
  api,
  type EtatFile,
  type RapportExport,
  type RapportImport,
  type SessionCapture,
} from "./api"

interface Props {
  onRetour: () => void
  surErreurAuth: (e: unknown) => boolean
}

export default function Technique({ onRetour, surErreurAuth }: Props) {
  const [sessions, setSessions] = useState<SessionCapture[] | null>(null)
  const [erreur, setErreur] = useState<string | null>(null)

  const surErreur = useCallback(
    (e: unknown) => {
      if (!surErreurAuth(e)) setErreur(e instanceof Error ? e.message : String(e))
    },
    [surErreurAuth],
  )

  const chargerSessions = useCallback(() => {
    api.sessions().then(setSessions).catch(surErreur)
  }, [surErreur])

  useEffect(chargerSessions, [chargerSessions])

  // ── Import ────────────────────────────────────────────────────────────────
  const [dossiers, setDossiers] = useState("")
  const [rattacher, setRattacher] = useState(false)
  const [nomSession, setNomSession] = useState("")
  const [dateCapture, setDateCapture] = useState("")
  const [eclairage, setEclairage] = useState("")
  const [hauteur, setHauteur] = useState("")
  const [etatCompost, setEtatCompost] = useState("")
  const [operateur, setOperateur] = useState("")
  const [notes, setNotes] = useState("")
  const [importEnCours, setImportEnCours] = useState(false)
  const [rapportImport, setRapportImport] = useState<RapportImport | null>(null)

  const lancerImport = async (e: FormEvent) => {
    e.preventDefault()
    const sources = dossiers
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.length > 0)
    if (sources.length === 0) return
    setErreur(null)
    setRapportImport(null)
    setImportEnCours(true)
    try {
      const rapport = await api.importer({
        source_dirs: sources,
        name: nomSession.trim(),
        rattacher,
        ...(rattacher
          ? {}
          : {
              captured_on: dateCapture,
              lighting: eclairage.trim() || undefined,
              camera_height_cm: hauteur.trim() ? Number(hauteur) : undefined,
              compost_state: etatCompost.trim() || undefined,
              operator: operateur.trim() || undefined,
              notes: notes.trim() || undefined,
            }),
      })
      setRapportImport(rapport)
      chargerSessions()
    } catch (err) {
      surErreur(err)
    } finally {
      setImportEnCours(false)
    }
  }

  // ── Export ────────────────────────────────────────────────────────────────
  const [toutExporter, setToutExporter] = useState(true)
  const [sessionsChoisies, setSessionsChoisies] = useState<Set<string>>(new Set())
  const [repertoireSortie, setRepertoireSortie] = useState("")
  const [exportEnCours, setExportEnCours] = useState(false)
  const [rapportExport, setRapportExport] = useState<RapportExport | null>(null)

  const basculerSession = (name: string) => {
    setSessionsChoisies((avant) => {
      const suivant = new Set(avant)
      if (suivant.has(name)) suivant.delete(name)
      else suivant.add(name)
      return suivant
    })
  }

  const lancerExport = async (e: FormEvent) => {
    e.preventDefault()
    setErreur(null)
    setRapportExport(null)
    setExportEnCours(true)
    try {
      setRapportExport(
        await api.exporter(
          repertoireSortie.trim(),
          toutExporter ? null : Array.from(sessionsChoisies),
        ),
      )
    } catch (err) {
      surErreur(err)
    } finally {
      setExportEnCours(false)
    }
  }

  // ── Pré-annotation ────────────────────────────────────────────────────────
  const [file, setFile] = useState<EtatFile | null>(null)

  const chargerFile = useCallback(() => {
    api.etatFile().then(setFile).catch(surErreur)
  }, [surErreur])

  useEffect(chargerFile, [chargerFile])

  const relancer = async (imageId: number) => {
    setErreur(null)
    try {
      await api.relancerGarees([imageId])
      chargerFile()
    } catch (err) {
      surErreur(err)
    }
  }

  const passerAAnnoter = async (imageId: number, nom: string) => {
    if (
      !window.confirm(
        `Passer « ${nom} » directement à « à annoter », sans pré-annotation ? ` +
          "Le worker ne la traitera plus.",
      )
    ) {
      return
    }
    setErreur(null)
    try {
      await api.passerAAnnoter([imageId])
      chargerFile()
    } catch (err) {
      surErreur(err)
    }
  }

  const paire = (liste: [string, string][], vide: string, fleche = " — ") =>
    liste.length === 0 ? (
      <p className="texte-doux">{vide}</p>
    ) : (
      <ul className="liste-rapport">
        {liste.map(([a, b], i) => (
          <li key={i}>
            <code>{a}</code>
            {fleche}
            {b}
          </li>
        ))}
      </ul>
    )

  return (
    <div className="page">
      <header className="barre">
        <button className="btn btn-petit" onClick={onRetour}>
          ← Lots
        </button>
        <h1>Administration technique</h1>
      </header>

      {erreur && <div className="bandeau bandeau-erreur">{erreur}</div>}

      {/* ── Import ─────────────────────────────────────────────────────── */}
      <h2>Import d'une session</h2>
      <div className="bandeau bandeau-brouillon">
        Plusieurs postes de capture simultanés forment <strong>une seule</strong> session : un
        dossier par poste, tous dans le même import (ou un rattachement ensuite). Deux sessions
        distinctes pour la même matière recréeraient la fuite entre entraînement et test.
      </div>
      <form className="carte-formulaire" onSubmit={lancerImport}>
        <label>
          Dossiers côté serveur (un par ligne, un par poste de capture)
          <textarea
            rows={3}
            value={dossiers}
            onChange={(e) => setDossiers(e.target.value)}
            placeholder={"/data/depot/2026-08-04_posteA\n/data/depot/2026-08-04_posteB"}
          />
        </label>
        <div className="rangee-champs">
          <label className="ligne-case">
            <input
              type="radio"
              checked={!rattacher}
              onChange={() => setRattacher(false)}
            />
            Créer une session
          </label>
          <label className="ligne-case">
            <input type="radio" checked={rattacher} onChange={() => setRattacher(true)} />
            Rattacher à une session existante
          </label>
        </div>
        {rattacher ? (
          <label>
            Session existante
            <select value={nomSession} onChange={(e) => setNomSession(e.target.value)}>
              <option value="">— choisir —</option>
              {(sessions ?? []).map((s) => (
                <option key={s.id} value={s.name}>
                  {s.name} ({s.captured_on}, {s.images} images)
                </option>
              ))}
            </select>
          </label>
        ) : (
          <>
            <div className="rangee-champs">
              <label>
                Nom de session
                <input
                  value={nomSession}
                  onChange={(e) => setNomSession(e.target.value)}
                  placeholder="2026-08-04_tas-A"
                />
              </label>
              <label>
                Date de capture
                <input
                  type="date"
                  value={dateCapture}
                  onChange={(e) => setDateCapture(e.target.value)}
                />
              </label>
            </div>
            <div className="rangee-champs">
              <label>
                Éclairage
                <input value={eclairage} onChange={(e) => setEclairage(e.target.value)} />
              </label>
              <label>
                Hauteur caméra (cm)
                <input
                  type="number"
                  min={0}
                  value={hauteur}
                  onChange={(e) => setHauteur(e.target.value)}
                />
              </label>
              <label>
                État du compost
                <input value={etatCompost} onChange={(e) => setEtatCompost(e.target.value)} />
              </label>
              <label>
                Opérateur
                <input value={operateur} onChange={(e) => setOperateur(e.target.value)} />
              </label>
            </div>
            <label>
              Notes
              <input value={notes} onChange={(e) => setNotes(e.target.value)} />
            </label>
          </>
        )}
        <button
          className="btn btn-primaire"
          disabled={
            importEnCours ||
            !dossiers.trim() ||
            !nomSession.trim() ||
            (!rattacher && !dateCapture)
          }
        >
          {importEnCours ? "Import en cours…" : "Importer"}
        </button>
      </form>
      {rapportImport && (
        <div className="carte-rapport">
          {rapportImport.aborted_reason ? (
            <p className="texte-erreur">Import refusé — {rapportImport.aborted_reason}</p>
          ) : (
            <p>
              <strong>{rapportImport.created.length}</strong> image(s) créée(s) dans «{" "}
              {rapportImport.session_name} ».
            </p>
          )}
          <h3>Doublons ignorés ({rapportImport.duplicates.length})</h3>
          {paire(rapportImport.duplicates, "aucun")}
          <h3>Fichiers rejetés ({rapportImport.rejected.length})</h3>
          {paire(rapportImport.rejected, "aucun")}
          <h3>Renommés pour l'export ({rapportImport.renamed.length})</h3>
          {paire(rapportImport.renamed, "aucun", " → ")}
        </div>
      )}

      {/* ── Export ─────────────────────────────────────────────────────── */}
      <h2>Export YOLO</h2>
      <form className="carte-formulaire" onSubmit={lancerExport}>
        <label className="ligne-case">
          <input
            type="checkbox"
            checked={toutExporter}
            onChange={(e) => setToutExporter(e.target.checked)}
          />
          Tout exporter (toutes les sessions)
        </label>
        {!toutExporter && (
          <div className="grille-sessions">
            {(sessions ?? []).map((s) => (
              <label key={s.id} className="ligne-case">
                <input
                  type="checkbox"
                  checked={sessionsChoisies.has(s.name)}
                  onChange={() => basculerSession(s.name)}
                />
                {s.name} ({s.images} images)
              </label>
            ))}
          </div>
        )}
        <label>
          Répertoire de sortie côté serveur
          <input
            value={repertoireSortie}
            onChange={(e) => setRepertoireSortie(e.target.value)}
            placeholder="/data/exports/2026-08-04"
          />
        </label>
        <button
          className="btn btn-primaire"
          disabled={
            exportEnCours ||
            !repertoireSortie.trim() ||
            (!toutExporter && sessionsChoisies.size === 0)
          }
        >
          {exportEnCours ? "Export en cours…" : "Exporter"}
        </button>
      </form>
      {rapportExport && (
        <div className="carte-rapport">
          <p>
            <strong>{rapportExport.images}</strong> images, <strong>{rapportExport.boxes}</strong>{" "}
            boîtes, {rapportExport.empty_labels} fichier(s) de labels vides (négatifs), vers{" "}
            <code>{rapportExport.output_dir}</code>.
          </p>
          <h3>Répartition par classe</h3>
          <ul className="liste-rapport">
            {Object.entries(rapportExport.class_counts).map(([classe, n]) => (
              <li key={classe}>
                {classe} : {n}
              </li>
            ))}
          </ul>
          <h3>Sessions couvertes</h3>
          <ul className="liste-rapport">
            {rapportExport.sessions.map((s) => (
              <li key={s.id}>
                {s.name} : {s.images} images, {s.boxes} boîtes
              </li>
            ))}
          </ul>
          {rapportExport.renamed.length > 0 && (
            <>
              <h3>Renommés ({rapportExport.renamed.length})</h3>
              {paire(rapportExport.renamed, "aucun", " → ")}
            </>
          )}
        </div>
      )}

      {/* ── Pré-annotation ─────────────────────────────────────────────── */}
      <h2>File de pré-annotation</h2>
      <p className="texte-doux">
        Le worker tourne à part (machine dédiée à terme) : cet écran observe sa file, il ne le
        pilote pas. Le court-circuit d'un lot entier reste sur l'écran des lots.
      </p>
      {file === null ? (
        <p className="texte-doux">Chargement…</p>
      ) : (
        <>
          <p>
            <strong>{file.en_attente}</strong> image(s) en attente de pré-annotation ·{" "}
            <strong>{file.garees.length}</strong> garée(s) (plafond :{" "}
            {file.plafond_tentatives} tentatives){" "}
            <button className="btn btn-petit" onClick={chargerFile}>
              Rafraîchir
            </button>
          </p>
          {file.garees.length > 0 && (
            <table className="table-lots">
              <thead>
                <tr>
                  <th>Image</th>
                  <th>Session · lot</th>
                  <th>Tentatives</th>
                  <th>Motif</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {file.garees.map((g) => (
                  <tr key={g.id}>
                    <td>
                      <code>{g.nom}</code>
                    </td>
                    <td>
                      {g.session} · {g.lot}
                    </td>
                    <td>{g.tentatives}</td>
                    <td>
                      {g.motif_type && <span className="badge badge-garees">{g.motif_type}</span>}{" "}
                      {g.motif}
                    </td>
                    <td className="cellule-actions">
                      <button
                        className="btn btn-petit"
                        title="Remettre le compteur de tentatives à zéro : l'image repasse dans la file du worker"
                        onClick={() => relancer(g.id)}
                      >
                        Remettre en file
                      </button>
                      <button
                        className="btn btn-petit"
                        title="Court-circuiter : passer à « à annoter » sans pré-annotation"
                        onClick={() => passerAAnnoter(g.id, g.nom)}
                      >
                        Passer à annoter
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
    </div>
  )
}
