/** Changement de mot de passe imposé à la première connexion : le mot de
 * passe initial a été posé par un administrateur (et a donc circulé), il doit
 * être remplacé avant d'accéder au reste. Le serveur révoque les autres
 * sessions du compte au passage. */
import { type FormEvent, useState } from "react"

import { api, type Moi } from "./api"

interface Props {
  moi: Moi
  onChange: (moi: Moi) => void
  onDeconnexion: () => void
}

export default function MotDePasse({ moi, onChange, onDeconnexion }: Props) {
  const [actuel, setActuel] = useState("")
  const [nouveau, setNouveau] = useState("")
  const [confirmation, setConfirmation] = useState("")
  const [erreur, setErreur] = useState<string | null>(null)
  const [enCours, setEnCours] = useState(false)

  const soumettre = async (e: FormEvent) => {
    e.preventDefault()
    if (nouveau !== confirmation) {
      setErreur("La confirmation ne correspond pas au nouveau mot de passe.")
      return
    }
    setErreur(null)
    setEnCours(true)
    try {
      onChange(await api.changerMotDePasse(actuel, nouveau))
    } catch (err) {
      setErreur(err instanceof Error ? err.message : "Changement impossible")
    } finally {
      setEnCours(false)
    }
  }

  return (
    <div className="login-fond">
      <form className="login-carte" onSubmit={soumettre}>
        <h1>Nouveau mot de passe</h1>
        <p className="texte-doux">
          Bonjour {moi.display_name ?? moi.username} — votre mot de passe a été
          posé par un administrateur : choisissez le vôtre pour continuer
          (8 caractères minimum).
        </p>
        <label>
          Mot de passe actuel
          <input
            autoFocus
            type="password"
            value={actuel}
            onChange={(e) => setActuel(e.target.value)}
            autoComplete="current-password"
          />
        </label>
        <label>
          Nouveau mot de passe
          <input
            type="password"
            value={nouveau}
            onChange={(e) => setNouveau(e.target.value)}
            autoComplete="new-password"
          />
        </label>
        <label>
          Confirmation
          <input
            type="password"
            value={confirmation}
            onChange={(e) => setConfirmation(e.target.value)}
            autoComplete="new-password"
          />
        </label>
        {erreur && <p className="texte-erreur">{erreur}</p>}
        <button
          className="btn btn-primaire"
          disabled={enCours || !actuel || nouveau.length < 8 || !confirmation}
        >
          {enCours ? "Changement…" : "Changer et continuer"}
        </button>
        <button type="button" className="btn btn-petit" onClick={onDeconnexion}>
          Se déconnecter
        </button>
      </form>
    </div>
  )
}
