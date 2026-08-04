import { type FormEvent, useState } from "react"

import { api, type Moi } from "./api"

export default function Login({ onConnecte }: { onConnecte: (moi: Moi) => void }) {
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [erreur, setErreur] = useState<string | null>(null)
  const [enCours, setEnCours] = useState(false)

  const soumettre = async (e: FormEvent) => {
    e.preventDefault()
    setErreur(null)
    setEnCours(true)
    try {
      onConnecte(await api.login(username, password))
    } catch (err) {
      setErreur(err instanceof Error ? err.message : "Connexion impossible")
    } finally {
      setEnCours(false)
    }
  }

  return (
    <div className="login-fond">
      <form className="login-carte" onSubmit={soumettre}>
        <h1>Compost — annotation</h1>
        <label>
          Compte
          <input
            autoFocus
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
          />
        </label>
        <label>
          Mot de passe
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
          />
        </label>
        {erreur && <p className="texte-erreur">{erreur}</p>}
        <button className="btn btn-primaire" disabled={enCours || !username || !password}>
          {enCours ? "Connexion…" : "Se connecter"}
        </button>
      </form>
    </div>
  )
}
