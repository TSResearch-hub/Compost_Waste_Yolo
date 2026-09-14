/** Matériel — flotte des cartes Jetson, administrateur. Déclaration d'une
 * carte (préalable à son premier envoi : le serveur refuse toute carte
 * inconnue ou désactivée), nom libre, mise hors service / remise en service.
 * Jamais de suppression : une carte désactivée garde ses sessions et leur
 * origine. L'identifiant est normalisé comme côté serveur (minuscules,
 * `:` → `-`) : une adresse MAC s'écrit indifféremment 48:B0:2D:3E:AA:01 ou
 * 48-b0-2d-3e-aa-01. */
import { type FormEvent, useCallback, useEffect, useState } from "react"

import { api, type Jetson, type Moi } from "./api"

// miroir de models.normaliser_jetson_id / JETSON_ID_REGEX : l'administrateur
// voit AVANT d'envoyer sous quelle forme la carte sera enregistrée, et une
// forme invalide est expliquée ici plutôt que par un 422 brut
const normaliserId = (brut: string) => brut.trim().toLowerCase().replace(/:/g, "-")
const ID_VALIDE = /^[a-z0-9_.-]{1,100}$/

const formatDate = (iso: string) =>
  new Date(iso).toLocaleString("fr-FR", { dateStyle: "short", timeStyle: "short" })

interface Props {
  moi: Moi
  onRetour: () => void
  surErreurAuth: (e: unknown) => boolean
}

export default function Flotte({ moi, onRetour, surErreurAuth }: Props) {
  const [cartes, setCartes] = useState<Jetson[] | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [erreur, setErreur] = useState<string | null>(null)

  // formulaire de déclaration
  const [identifiant, setIdentifiant] = useState("")
  const [nom, setNom] = useState("")
  const [enCreation, setEnCreation] = useState(false)

  const charger = useCallback(() => {
    api
      .jetsons()
      .then(setCartes)
      .catch((e) => {
        if (!surErreurAuth(e)) setErreur(e instanceof Error ? e.message : String(e))
      })
  }, [surErreurAuth])

  useEffect(charger, [charger])

  // exécute une action serveur puis recharge la liste ; vrai si elle a abouti
  const agir = async (action: () => Promise<unknown>, succes: string): Promise<boolean> => {
    setMessage(null)
    setErreur(null)
    try {
      await action()
      setMessage(succes)
      charger()
      return true
    } catch (e) {
      if (!surErreurAuth(e)) setErreur(e instanceof Error ? e.message : String(e))
      return false
    }
  }

  const idNormalise = normaliserId(identifiant)
  const idValide = ID_VALIDE.test(idNormalise)

  const declarer = async (e: FormEvent) => {
    e.preventDefault()
    if (!idValide) {
      setMessage(null)
      setErreur(
        "Identifiant invalide : lettres, chiffres, « . », « _ », « - » uniquement " +
          "(une adresse MAC convient), 100 caractères maximum.",
      )
      return
    }
    setEnCreation(true)
    const ok = await agir(
      () => api.creerJetson(idNormalise, nom.trim() || null),
      `Carte « ${idNormalise} » déclarée — elle peut désormais envoyer ses sessions.`,
    )
    setEnCreation(false)
    // en cas de refus (identifiant déjà pris…) la saisie reste, à corriger
    if (ok) {
      setIdentifiant("")
      setNom("")
    }
  }

  const renommer = (c: Jetson) => {
    const nouveau = window.prompt(`Nom de la carte « ${c.id} » (vide pour l'effacer) :`, c.name ?? "")
    if (nouveau === null) return
    const name = nouveau.trim() || null
    if (name === c.name) return
    void agir(
      () => api.modifierJetson(c.id, { name }),
      `« ${c.id} » : nom ${name === null ? "effacé" : `changé en « ${name} »`}.`,
    )
  }

  const basculerActive = (c: Jetson) => {
    if (
      c.is_active &&
      !window.confirm(
        `Désactiver la carte « ${c.id} » ? Ses envois seront refusés jusqu'à sa remise en ` +
          "service ; ses sessions déjà reçues restent.",
      )
    ) {
      return
    }
    void agir(
      () => api.modifierJetson(c.id, { is_active: !c.is_active }),
      `« ${c.id} » ${c.is_active ? "désactivée : ses envois sont refusés" : "remise en service"}.`,
    )
  }

  if (cartes === null) {
    return <div className="plein-ecran-message">{erreur ?? "Chargement de la flotte…"}</div>
  }

  const actives = cartes.filter((c) => c.is_active).length

  return (
    <div className="page">
      <header className="barre">
        <button className="btn btn-petit" onClick={onRetour}>
          ← Lots
        </button>
        <h1>Matériel</h1>
        <div className="barre-droite">
          <span className="texte-doux">
            {moi.display_name ?? moi.username} · {moi.role}
          </span>
        </div>
      </header>

      {message && <div className="bandeau bandeau-info">{message}</div>}
      {erreur && <div className="bandeau bandeau-erreur">{erreur}</div>}

      <h2>Nouvelle carte Jetson</h2>
      <form className="formulaire-ligne" onSubmit={declarer}>
        <label>
          Identifiant
          <input
            value={identifiant}
            onChange={(e) => setIdentifiant(e.target.value)}
            placeholder="adresse MAC, ex. 48:b0:2d:3e:aa:01"
            autoComplete="off"
            spellCheck={false}
          />
        </label>
        <label>
          Nom
          <input
            value={nom}
            onChange={(e) => setNom(e.target.value)}
            placeholder="facultatif, ex. Tas A"
          />
        </label>
        <button className="btn btn-primaire" disabled={enCreation || !idValide}>
          Déclarer
        </button>
        {identifiant.trim() !== "" && (
          <span className="texte-doux">
            {idValide
              ? idNormalise !== identifiant
                ? `sera enregistrée comme « ${idNormalise} »`
                : ""
              : "identifiant invalide (lettres, chiffres, « . », « _ », « - », 100 max.)"}
          </span>
        )}
      </form>
      <p className="texte-doux">
        Une carte doit être déclarée ici avant son premier envoi : le serveur refuse toute
        session venant d'une carte inconnue ou désactivée. Une carte ne se supprime jamais —
        désactivez-la, ses sessions et leur origine survivent.
      </p>

      <h2>
        Cartes déclarées{" "}
        <span className="texte-doux">
          ({actives} en service / {cartes.length})
        </span>
      </h2>
      <table className="table-lots">
        <thead>
          <tr>
            <th>Identifiant</th>
            <th>Nom</th>
            <th>État</th>
            <th>Déclarée le</th>
            <th>Modifiée le</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {cartes.length === 0 && (
            <tr>
              <td colSpan={6} className="texte-doux">
                Aucune carte déclarée.
              </td>
            </tr>
          )}
          {cartes.map((c) => (
            <tr key={c.id} className={c.is_active ? "" : "ligne-inactive"}>
              <td>
                <code>{c.id}</code>
              </td>
              <td>
                {c.name ?? <span className="texte-doux">—</span>}{" "}
                <button className="btn btn-petit" onClick={() => renommer(c)}>
                  Renommer
                </button>
              </td>
              <td>
                {c.is_active ? (
                  <span className="badge badge-validee">en service</span>
                ) : (
                  <span className="badge badge-rejetee" title="Ses envois sont refusés">
                    désactivée
                  </span>
                )}
              </td>
              <td>{formatDate(c.created_at)}</td>
              <td>{formatDate(c.updated_at)}</td>
              <td className="cellule-actions">
                <button
                  className={`btn btn-petit${c.is_active ? " btn-danger" : ""}`}
                  title={
                    c.is_active
                      ? "Refuser ses envois — ses sessions déjà reçues restent"
                      : "Accepter à nouveau ses envois"
                  }
                  onClick={() => basculerActive(c)}
                >
                  {c.is_active ? "Désactiver" : "Remettre en service"}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
