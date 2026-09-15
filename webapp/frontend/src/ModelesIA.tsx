/** Modèles IA — publication des poids pour la flotte Jetson, administrateur.
 * Un nom de version, le `.pt` et le `data.yaml` de l'entraînement : le
 * serveur les range sous `models/{version}/` et la version passe en tête de
 * l'historique — c'est celle que les cartes récupèrent (GET
 * /api/models_ia/latest avec leur jeton, hors de cet écran). Une version ne
 * se modifie ni ne se supprime : republier = publier sous un nouveau nom. */
import { type FormEvent, useCallback, useEffect, useState } from "react"

import { api, type ModelVersion, type Moi } from "./api"

// miroir de models.MODEL_VERSION_NAME_REGEX : le nom nomme un dossier du
// stockage — une forme invalide est expliquée ici plutôt que par un 422 brut
const NOM_VALIDE = /^[A-Za-z0-9_.-]{1,100}$/
const EXT_PT = /\.pt$/i
const EXT_YAML = /\.ya?ml$/i

const formatDate = (iso: string) =>
  new Date(iso).toLocaleString("fr-FR", { dateStyle: "short", timeStyle: "short" })

const formatTaille = (octets: number) =>
  octets >= 1_048_576 ? `${(octets / 1_048_576).toFixed(1)} Mo` : `${Math.ceil(octets / 1024)} ko`

interface Props {
  moi: Moi
  onRetour: () => void
  surErreurAuth: (e: unknown) => boolean
}

export default function ModelesIA({ moi, onRetour, surErreurAuth }: Props) {
  const [versions, setVersions] = useState<ModelVersion[] | null>(null)
  // id → nom affiché, pour la colonne « Par » ; un simple confort : sans la
  // liste des comptes, l'id suffit
  const [auteurs, setAuteurs] = useState<Record<number, string>>({})
  const [message, setMessage] = useState<string | null>(null)
  const [erreur, setErreur] = useState<string | null>(null)

  // formulaire de publication
  const [nomVersion, setNomVersion] = useState("")
  const [pt, setPt] = useState<File | null>(null)
  const [yaml, setYaml] = useState<File | null>(null)
  const [enEnvoi, setEnEnvoi] = useState(false)
  // la valeur d'un <input type="file"> n'est pas contrôlable : après un envoi
  // réussi on remonte le formulaire (clé) pour le vider proprement
  const [cleFormulaire, setCleFormulaire] = useState(0)

  const charger = useCallback(() => {
    api
      .listerModeles()
      .then(setVersions)
      .catch((e) => {
        if (!surErreurAuth(e)) setErreur(e instanceof Error ? e.message : String(e))
      })
  }, [surErreurAuth])

  useEffect(charger, [charger])

  useEffect(() => {
    api
      .utilisateurs()
      .then((tous) =>
        setAuteurs(Object.fromEntries(tous.map((u) => [u.id, u.display_name ?? u.username]))),
      )
      .catch(() => {
        /* confort seulement — la colonne affichera l'id */
      })
  }, [])

  const nom = nomVersion.trim()
  const nomValide = NOM_VALIDE.test(nom)
  const ptValide = pt !== null && EXT_PT.test(pt.name)
  const yamlValide = yaml !== null && EXT_YAML.test(yaml.name)
  const pret = nomValide && ptValide && yamlValide && !enEnvoi
  const dejaPrise = versions?.some((v) => v.version_name === nom) ?? false

  const deployer = async (e: FormEvent) => {
    e.preventDefault()
    if (!pret || pt === null || yaml === null) return
    if (
      !window.confirm(
        `Déployer « ${nom} » sur la flotte ?\n` +
          "Dès sa publication, c'est cette version que les cartes récupéreront. " +
          "Une version publiée ne se modifie ni ne se supprime.",
      )
    ) {
      return
    }
    setMessage(null)
    setErreur(null)
    setEnEnvoi(true)
    try {
      const v = await api.uploadModele(nom, pt, yaml)
      setMessage(
        `Version « ${v.version_name} » publiée — c'est désormais celle que les cartes récupèrent.`,
      )
      setNomVersion("")
      setPt(null)
      setYaml(null)
      setCleFormulaire((k) => k + 1)
      charger()
    } catch (e) {
      // en cas de refus (nom déjà pris, data.yaml invalide…) la saisie
      // reste, à corriger
      if (!surErreurAuth(e)) setErreur(e instanceof Error ? e.message : String(e))
    } finally {
      setEnEnvoi(false)
    }
  }

  if (versions === null) {
    return <div className="plein-ecran-message">{erreur ?? "Chargement des modèles…"}</div>
  }

  // ce que le formulaire attend encore — un seul message, le plus utile
  let indication: string | null = null
  if (nom !== "" && !nomValide) {
    indication = "nom invalide : lettres, chiffres, « . », « _ », « - » uniquement, 100 max."
  } else if (nomValide && dejaPrise) {
    indication = `« ${nom} » existe déjà — une version ne se remplace pas, choisissez un autre nom.`
  } else if (pt !== null && !ptValide) {
    indication = "le fichier de poids doit porter l'extension .pt"
  } else if (yaml !== null && !yamlValide) {
    indication = "le fichier de classes doit porter l'extension .yaml ou .yml"
  } else if (pt !== null && yaml !== null) {
    indication = `${pt.name} (${formatTaille(pt.size)}) · ${yaml.name} (${formatTaille(yaml.size)})`
  }

  return (
    <div className="page">
      <header className="barre">
        <button className="btn btn-petit" onClick={onRetour}>
          ← Lots
        </button>
        <h1>Modèles IA</h1>
        <div className="barre-droite">
          <span className="texte-doux">
            {moi.display_name ?? moi.username} · {moi.role}
          </span>
        </div>
      </header>

      {message && <div className="bandeau bandeau-info">{message}</div>}
      {erreur && <div className="bandeau bandeau-erreur">{erreur}</div>}

      <h2>Publier une nouvelle version</h2>
      <form key={cleFormulaire} className="formulaire-ligne" onSubmit={deployer}>
        <label>
          Nom de version
          <input
            value={nomVersion}
            onChange={(e) => setNomVersion(e.target.value)}
            placeholder="ex. v2.1-caisson"
            autoComplete="off"
            spellCheck={false}
            disabled={enEnvoi}
          />
        </label>
        <label>
          Poids (.pt)
          <input
            type="file"
            accept=".pt"
            onChange={(e) => setPt(e.target.files?.[0] ?? null)}
            disabled={enEnvoi}
          />
        </label>
        <label>
          Classes (data.yaml)
          <input
            type="file"
            accept=".yaml,.yml"
            onChange={(e) => setYaml(e.target.files?.[0] ?? null)}
            disabled={enEnvoi}
          />
        </label>
        <button className="btn btn-primaire" disabled={!pret || dejaPrise}>
          {enEnvoi ? "Envoi en cours…" : "Déployer sur la flotte"}
        </button>
        {indication && <span className="texte-doux">{indication}</span>}
      </form>
      <p className="texte-doux">
        Le <code>.pt</code> et le <code>data.yaml</code> de l'entraînement vont ensemble : les
        classes d'un modèle sont celles avec lesquelles il a été entraîné. Les cartes Jetson
        interrogent le serveur avec leur jeton et récupèrent la version en tête de
        l'historique ; une version publiée ne se modifie ni ne se supprime — pour corriger,
        publiez sous un nouveau nom.
      </p>

      <h2>
        Historique des versions{" "}
        <span className="texte-doux">
          ({versions.length} publiée{versions.length > 1 ? "s" : ""})
        </span>
      </h2>
      <table className="table-lots">
        <thead>
          <tr>
            <th>Version</th>
            <th>État</th>
            <th>Publiée le</th>
            <th>Par</th>
            <th>Emplacement</th>
          </tr>
        </thead>
        <tbody>
          {versions.length === 0 && (
            <tr>
              <td colSpan={5} className="texte-doux">
                Aucune version publiée — les cartes n'ont encore aucun modèle à récupérer.
              </td>
            </tr>
          )}
          {versions.map((v, i) => (
            <tr key={v.id} className={i === 0 ? "ligne-courante" : ""}>
              <td>
                <strong>{v.version_name}</strong>
              </td>
              <td>
                {i === 0 ? (
                  <span
                    className="badge badge-validee"
                    title="La version que les cartes récupèrent (GET /api/models_ia/latest)"
                  >
                    Version actuelle en production
                  </span>
                ) : (
                  <span className="badge badge-chrono">précédente</span>
                )}
              </td>
              <td>{formatDate(v.created_at)}</td>
              <td>{auteurs[v.created_by] ?? `compte #${v.created_by}`}</td>
              <td className="texte-doux">
                <code>{v.pt_file_path}</code>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
