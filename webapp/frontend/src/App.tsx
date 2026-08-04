/** Aiguillage : login → (changement de mot de passe imposé) → lots →
 * annotation, plus les écrans administrateur (comptes, technique). Pas de
 * routeur — des vues, un état. La session vit dans le cookie HttpOnly ; au
 * chargement on demande simplement au serveur qui on est. */
import { useCallback, useEffect, useState } from "react"

import { api, ApiError, type Moi } from "./api"
import Annotation from "./Annotation"
import Comptes from "./Comptes"
import Login from "./Login"
import Lots from "./Lots"
import MotDePasse from "./MotDePasse"
import Technique from "./Technique"

type Vue =
  | { nom: "chargement" }
  | { nom: "login" }
  | { nom: "lots" }
  | { nom: "annotation"; lotId: number; lotNom: string }
  | { nom: "comptes" }
  | { nom: "technique" }

export default function App() {
  const [moi, setMoi] = useState<Moi | null>(null)
  const [vue, setVue] = useState<Vue>({ nom: "chargement" })

  useEffect(() => {
    api
      .moi()
      .then((m) => {
        setMoi(m)
        setVue({ nom: "lots" })
      })
      .catch(() => setVue({ nom: "login" }))
  }, [])

  const deconnexion = useCallback(async () => {
    try {
      await api.logout()
    } catch {
      /* la session a pu expirer : on retourne au login quoi qu'il arrive */
    }
    setMoi(null)
    setVue({ nom: "login" })
  }, [])

  // Une 401 en cours de route (session expirée) ramène au login
  const surErreurAuth = useCallback((e: unknown) => {
    if (e instanceof ApiError && e.status === 401) {
      setMoi(null)
      setVue({ nom: "login" })
      return true
    }
    return false
  }, [])

  if (vue.nom === "chargement") {
    return <div className="plein-ecran-message">Chargement…</div>
  }
  if (vue.nom === "login" || moi === null) {
    return (
      <Login
        onConnecte={(m) => {
          setMoi(m)
          setVue({ nom: "lots" })
        }}
      />
    )
  }
  // mot de passe posé par un administrateur : à remplacer avant tout le reste
  if (moi.must_change_password) {
    return <MotDePasse moi={moi} onChange={setMoi} onDeconnexion={deconnexion} />
  }
  if (vue.nom === "lots") {
    return (
      <Lots
        moi={moi}
        onOuvrirLot={(lotId, lotNom) => setVue({ nom: "annotation", lotId, lotNom })}
        onOuvrirComptes={() => setVue({ nom: "comptes" })}
        onOuvrirTechnique={() => setVue({ nom: "technique" })}
        onDeconnexion={deconnexion}
        surErreurAuth={surErreurAuth}
      />
    )
  }
  if (vue.nom === "comptes") {
    return <Comptes moi={moi} onRetour={() => setVue({ nom: "lots" })} surErreurAuth={surErreurAuth} />
  }
  if (vue.nom === "technique") {
    return <Technique onRetour={() => setVue({ nom: "lots" })} surErreurAuth={surErreurAuth} />
  }
  return (
    <Annotation
      moi={moi}
      lotId={vue.lotId}
      lotNom={vue.lotNom}
      onRetour={() => setVue({ nom: "lots" })}
      surErreurAuth={surErreurAuth}
    />
  )
}
