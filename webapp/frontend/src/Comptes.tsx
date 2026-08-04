/** Gestion des comptes — administrateur. Création, rôle, nom affiché,
 * désactivation/réactivation (jamais de suppression : la traçabilité doit
 * survivre), réinitialisation de mot de passe. Tout mot de passe posé ici
 * devra être remplacé par l'intéressé à sa prochaine connexion. La garde
 * anti-verrouillage du serveur est reflétée à l'écran : un administrateur ne
 * modifie ni son propre rôle ni son propre statut. */
import { type FormEvent, useCallback, useEffect, useState } from "react"

import { api, type Moi, type Utilisateur } from "./api"

const ROLES = ["annotateur", "annotateur_confirme", "administrateur"] as const

const LIBELLES_ROLE: Record<string, string> = {
  annotateur: "annotateur",
  annotateur_confirme: "annotateur confirmé",
  administrateur: "administrateur",
}

interface Props {
  moi: Moi
  onRetour: () => void
  surErreurAuth: (e: unknown) => boolean
}

export default function Comptes({ moi, onRetour, surErreurAuth }: Props) {
  const [comptes, setComptes] = useState<Utilisateur[] | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [erreur, setErreur] = useState<string | null>(null)

  // formulaire de création
  const [identifiant, setIdentifiant] = useState("")
  const [nomAffiche, setNomAffiche] = useState("")
  const [role, setRole] = useState<string>("annotateur")
  const [motDePasse, setMotDePasse] = useState("")
  const [enCreation, setEnCreation] = useState(false)

  const charger = useCallback(() => {
    api
      .utilisateurs()
      .then(setComptes)
      .catch((e) => {
        if (!surErreurAuth(e)) setErreur(e instanceof Error ? e.message : String(e))
      })
  }, [surErreurAuth])

  useEffect(charger, [charger])

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

  const creer = async (e: FormEvent) => {
    e.preventDefault()
    setEnCreation(true)
    await agir(
      () =>
        api.creerUtilisateur({
          username: identifiant.trim(),
          password: motDePasse,
          display_name: nomAffiche.trim() || null,
          role,
        }),
      `Compte « ${identifiant.trim()} » créé — il devra changer son mot de passe à sa première connexion.`,
    )
    setEnCreation(false)
    setIdentifiant("")
    setNomAffiche("")
    setMotDePasse("")
  }

  const changerRole = (u: Utilisateur, nouveau: string) => {
    if (nouveau === u.role) return
    void agir(
      () => api.modifierUtilisateur(u.id, { role: nouveau }),
      `« ${u.username} » : rôle changé en ${LIBELLES_ROLE[nouveau] ?? nouveau}.`,
    )
  }

  const renommer = (u: Utilisateur) => {
    const nouveau = window.prompt(
      `Nom affiché de « ${u.username} » (vide pour l'effacer) :`,
      u.display_name ?? "",
    )
    if (nouveau === null) return
    void agir(
      () => api.modifierUtilisateur(u.id, { display_name: nouveau.trim() || null }),
      `« ${u.username} » : nom affiché mis à jour.`,
    )
  }

  const basculerActif = (u: Utilisateur) => {
    const verbe = u.is_active ? "Désactiver" : "Réactiver"
    if (
      !window.confirm(
        `${verbe} le compte « ${u.username} » ?` +
          (u.is_active ? " Ses sessions ouvertes seront immédiatement invalides." : ""),
      )
    ) {
      return
    }
    void agir(
      () => api.modifierUtilisateur(u.id, { is_active: !u.is_active }),
      `« ${u.username} » ${u.is_active ? "désactivé" : "réactivé"}.`,
    )
  }

  const reinitialiser = (u: Utilisateur) => {
    const nouveau = window.prompt(
      `Nouveau mot de passe pour « ${u.username} » (8 caractères minimum) — ` +
        "il devra le remplacer à sa prochaine connexion :",
    )
    if (nouveau === null) return
    if (nouveau.length < 8) {
      setErreur("Mot de passe trop court : 8 caractères minimum.")
      return
    }
    void agir(
      () => api.modifierUtilisateur(u.id, { password: nouveau }),
      `Mot de passe de « ${u.username} » réinitialisé — changement obligatoire à sa prochaine connexion.`,
    )
  }

  if (comptes === null) {
    return <div className="plein-ecran-message">{erreur ?? "Chargement des comptes…"}</div>
  }

  return (
    <div className="page">
      <header className="barre">
        <button className="btn btn-petit" onClick={onRetour}>
          ← Lots
        </button>
        <h1>Comptes</h1>
        <div className="barre-droite">
          <span className="texte-doux">
            {moi.display_name ?? moi.username} · {LIBELLES_ROLE[moi.role] ?? moi.role}
          </span>
        </div>
      </header>

      {message && <div className="bandeau bandeau-info">{message}</div>}
      {erreur && <div className="bandeau bandeau-erreur">{erreur}</div>}

      <h2>Nouveau compte</h2>
      <form className="formulaire-ligne" onSubmit={creer}>
        <label>
          Identifiant
          <input value={identifiant} onChange={(e) => setIdentifiant(e.target.value)} />
        </label>
        <label>
          Nom affiché
          <input
            value={nomAffiche}
            onChange={(e) => setNomAffiche(e.target.value)}
            placeholder="facultatif"
          />
        </label>
        <label>
          Rôle
          <select value={role} onChange={(e) => setRole(e.target.value)}>
            {ROLES.map((r) => (
              <option key={r} value={r}>
                {LIBELLES_ROLE[r]}
              </option>
            ))}
          </select>
        </label>
        <label>
          Mot de passe initial
          <input
            type="text"
            value={motDePasse}
            onChange={(e) => setMotDePasse(e.target.value)}
            placeholder="8 caractères min."
            autoComplete="off"
          />
        </label>
        <button
          className="btn btn-primaire"
          disabled={enCreation || !identifiant.trim() || motDePasse.length < 8}
        >
          Créer
        </button>
      </form>
      <p className="texte-doux">
        Le mot de passe initial est à communiquer à l'intéressé : il devra le remplacer à sa
        première connexion. Un compte ne se supprime jamais — désactivez-le, la traçabilité de
        ses annotations survit.
      </p>

      <h2>Comptes existants</h2>
      <table className="table-lots">
        <thead>
          <tr>
            <th>Identifiant</th>
            <th>Nom affiché</th>
            <th>Rôle</th>
            <th>État</th>
            <th>Créé le</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {comptes.map((u) => {
            const moiMeme = u.id === moi.id
            return (
              <tr key={u.id} className={u.is_active ? "" : "ligne-inactive"}>
                <td>
                  {u.username}
                  {moiMeme && <span className="texte-doux"> (vous)</span>}
                </td>
                <td>
                  {u.display_name ?? <span className="texte-doux">—</span>}{" "}
                  <button className="btn btn-petit" onClick={() => renommer(u)}>
                    Renommer
                  </button>
                </td>
                <td>
                  <select
                    value={u.role}
                    disabled={moiMeme}
                    title={
                      moiMeme
                        ? "Un administrateur ne modifie pas son propre rôle (anti-verrouillage)"
                        : undefined
                    }
                    onChange={(e) => changerRole(u, e.target.value)}
                  >
                    {ROLES.map((r) => (
                      <option key={r} value={r}>
                        {LIBELLES_ROLE[r]}
                      </option>
                    ))}
                  </select>
                </td>
                <td>
                  {u.is_active ? (
                    <span className="badge badge-validee">actif</span>
                  ) : (
                    <span className="badge badge-rejetee">désactivé</span>
                  )}
                  {u.must_change_password && (
                    <span
                      className="badge badge-garees"
                      title="Mot de passe posé par un administrateur : changement obligatoire à sa prochaine connexion"
                    >
                      mdp à changer
                    </span>
                  )}
                </td>
                <td>{new Date(u.created_at).toLocaleDateString("fr-FR")}</td>
                <td className="cellule-actions">
                  <button
                    className="btn btn-petit"
                    onClick={() => reinitialiser(u)}
                    title="Poser un nouveau mot de passe — l'intéressé devra le remplacer à sa prochaine connexion"
                  >
                    Réinitialiser le mdp
                  </button>
                  <button
                    className={`btn btn-petit${u.is_active ? " btn-danger" : ""}`}
                    disabled={moiMeme}
                    title={
                      moiMeme
                        ? "Un administrateur ne modifie pas son propre statut (anti-verrouillage)"
                        : undefined
                    }
                    onClick={() => basculerActif(u)}
                  >
                    {u.is_active ? "Désactiver" : "Réactiver"}
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
