/** Dataset — récupérer les données d'entraînement et déposer des images
 * brutes depuis son PC, administrateur. Deux zones :
 * - télécharger : un ZIP construit à la volée par le serveur (GET
 *   /api/dataset/export) — images annotées, labels YOLO des boîtes validées,
 *   data.yaml — même périmètre que l'export Technique, mais sans chemin côté
 *   serveur : le fichier arrive dans le navigateur ;
 * - importer : un ZIP d'images brutes (POST /api/dataset/import) qui rejoint
 *   la session « Import_Manuel_Admin » et passe dans la file de
 *   pré-annotation, comme un envoi de carte Jetson. */
import { type FormEvent, useCallback, useEffect, useState } from "react"

import { api, type Moi, type RapportImport, type ResumeDataset } from "./api"

const EXT_ZIP = /\.zip$/i

const formatTaille = (octets: number) =>
  octets >= 1_048_576 ? `${(octets / 1_048_576).toFixed(1)} Mo` : `${Math.ceil(octets / 1024)} ko`

interface Props {
  moi: Moi
  onRetour: () => void
  surErreurAuth: (e: unknown) => boolean
}

export default function Dataset({ moi, onRetour, surErreurAuth }: Props) {
  // ce que l'export contiendra — chargé à l'ouverture, rechargé après un dépôt
  const [resume, setResume] = useState<ResumeDataset | null>(null)
  const [erreurResume, setErreurResume] = useState<string | null>(null)

  // dépôt d'images brutes
  const [archive, setArchive] = useState<File | null>(null)
  const [enEnvoi, setEnEnvoi] = useState(false)
  const [rapport, setRapport] = useState<RapportImport | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [erreur, setErreur] = useState<string | null>(null)
  // la valeur d'un <input type="file"> n'est pas contrôlable : après un envoi
  // réussi on remonte le formulaire (clé) pour le vider proprement
  const [cleFormulaire, setCleFormulaire] = useState(0)

  const charger = useCallback(() => {
    api
      .resumeDataset()
      .then((r) => {
        setResume(r)
        setErreurResume(null)
      })
      .catch((e) => {
        if (!surErreurAuth(e)) setErreurResume(e instanceof Error ? e.message : String(e))
      })
  }, [surErreurAuth])

  useEffect(charger, [charger])

  // Pas de fetch : le navigateur suit l'URL avec le cookie de session et écrit
  // le fichier sur disque au fil de l'eau (Content-Disposition: attachment —
  // la page reste affichée). Un ZIP de plusieurs centaines de Mo ne doit pas
  // transiter par la mémoire de la page.
  const telecharger = () => {
    window.location.assign(api.exportDatasetUrl)
  }

  const archiveValide = archive !== null && EXT_ZIP.test(archive.name)
  const pret = archiveValide && !enEnvoi

  const envoyer = async (e: FormEvent) => {
    e.preventDefault()
    if (!pret || archive === null) return
    setMessage(null)
    setErreur(null)
    setRapport(null)
    setEnEnvoi(true)
    try {
      const r = await api.importerDatasetZip(archive)
      setRapport(r)
      if (r.aborted_reason) {
        setErreur(`Import refusé — ${r.aborted_reason}`)
      } else {
        setMessage(
          `${r.created.length} image(s) ajoutée(s) à « ${r.session_name} » : ` +
            "elles passent dans la file de pré-annotation.",
        )
        setArchive(null)
        setCleFormulaire((k) => k + 1)
      }
      charger()
    } catch (e) {
      // refus (archive illisible, trop grosse…) : la sélection reste, à corriger
      if (!surErreurAuth(e)) setErreur(e instanceof Error ? e.message : String(e))
    } finally {
      setEnEnvoi(false)
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

  let indication: string | null = null
  if (archive !== null && !archiveValide) {
    indication = "le fichier doit porter l'extension .zip"
  } else if (archive !== null) {
    indication = `${archive.name} (${formatTaille(archive.size)})`
  }

  const classesRenseignees = resume
    ? Object.entries(resume.class_counts).filter(([, n]) => n > 0)
    : []

  return (
    <div className="page">
      <header className="barre">
        <button className="btn btn-petit" onClick={onRetour}>
          ← Lots
        </button>
        <h1>Dataset</h1>
        <div className="barre-droite">
          <span className="texte-doux">
            {moi.display_name ?? moi.username} · {moi.role}
          </span>
        </div>
      </header>

      {message && <div className="bandeau bandeau-info">{message}</div>}
      {erreur && <div className="bandeau bandeau-erreur">{erreur}</div>}

      {/* ── Télécharger ───────────────────────────────────────────────── */}
      <h2>Télécharger le dataset</h2>
      <div className="carte-formulaire">
        <p className="texte-doux">
          Toutes les images annotées ou relues, avec leurs boîtes validées, au format YOLO :{" "}
          <code>images/</code>, <code>labels/</code> (un <code>.txt</code> par image, vide pour
          une image sans intrus) et <code>data.yaml</code> (les classes). L'archive contient
          aussi <code>groups.csv</code> et <code>classes.txt</code> : décompressée, elle se
          donne telle quelle à <code>prepare_dataset.py</code> de compost-yolo.
        </p>
        {erreurResume ? (
          <p className="texte-erreur">Export impossible — {erreurResume}</p>
        ) : resume === null ? (
          <p className="texte-doux">Chargement…</p>
        ) : (
          <p>
            <strong>{resume.images}</strong> image{resume.images > 1 ? "s" : ""} annotée
            {resume.images > 1 ? "s" : ""} · <strong>{resume.boxes}</strong> boîte
            {resume.boxes > 1 ? "s" : ""} validée{resume.boxes > 1 ? "s" : ""} ·{" "}
            {resume.empty_labels} négatif{resume.empty_labels > 1 ? "s" : ""} ·{" "}
            {resume.sessions.length} session{resume.sessions.length > 1 ? "s" : ""}
            {classesRenseignees.length > 0 && (
              <span className="texte-doux">
                {" "}
                ({classesRenseignees.map(([classe, n]) => `${classe} : ${n}`).join(", ")})
              </span>
            )}
            {resume.fichiers_manquants > 0 && (
              <>
                <br />
                <span className="texte-erreur">
                  {resume.fichiers_manquants} image{resume.fichiers_manquants > 1 ? "s" : ""}{" "}
                  annotée{resume.fichiers_manquants > 1 ? "s" : ""} dont le fichier est absent du
                  stockage : ignorée{resume.fichiers_manquants > 1 ? "s" : ""} (voir{" "}
                  <code>rapport.txt</code> dans l'archive).
                </span>
              </>
            )}
          </p>
        )}
        <button
          className="btn btn-primaire btn-large"
          onClick={telecharger}
          disabled={resume === null || resume.images === 0 || erreurResume !== null}
          title={
            resume !== null && resume.images === 0
              ? "Aucune image annotée pour l'instant"
              : "Télécharger l'archive ZIP"
          }
        >
          ⬇ Télécharger le dataset (ZIP)
        </button>
      </div>

      {/* ── Importer ──────────────────────────────────────────────────── */}
      <h2>Importer des images brutes</h2>
      <form key={cleFormulaire} className="carte-formulaire" onSubmit={envoyer}>
        <p className="texte-doux">
          Un fichier <code>.zip</code> contenant des images (<code>.jpg</code>,{" "}
          <code>.jpeg</code>, <code>.png</code>), sans annotation. Elles rejoignent la session
          « Import_Manuel_Admin » (lot « import ») et passent dans la file de
          pré-annotation ; une image déjà en base est ignorée, les autres fichiers sont
          rejetés. Toutes les images déposées ici partagent la même session, donc le même côté
          du découpage entraînement / test.
        </p>
        <div className="rangee-champs">
          <label>
            Archive (.zip)
            <input
              type="file"
              accept=".zip,application/zip"
              onChange={(e) => setArchive(e.target.files?.[0] ?? null)}
              disabled={enEnvoi}
            />
          </label>
        </div>
        <button className="btn btn-primaire" disabled={!pret}>
          {enEnvoi ? "Envoi en cours…" : "Envoyer"}
        </button>
        {indication && <span className="texte-doux">{indication}</span>}
      </form>
      {rapport && (
        <div className="carte-rapport">
          {rapport.aborted_reason ? (
            <p className="texte-erreur">Import refusé — {rapport.aborted_reason}</p>
          ) : (
            <p>
              <strong>{rapport.created.length}</strong> image(s) créée(s) dans «{" "}
              {rapport.session_name} ».
            </p>
          )}
          <h3>Doublons ignorés ({rapport.duplicates.length})</h3>
          {paire(rapport.duplicates, "aucun")}
          <h3>Fichiers rejetés ({rapport.rejected.length})</h3>
          {paire(rapport.rejected, "aucun")}
          <h3>Renommés pour l'export ({rapport.renamed.length})</h3>
          {paire(rapport.renamed, "aucun", " → ")}
        </div>
      )}
    </div>
  )
}
