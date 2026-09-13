/** État React persisté dans le localStorage du navigateur.
 *
 * Sert aux préférences d'affichage des annotateurs (opacité, épaisseur…) :
 * la valeur survit au rechargement et d'une session à l'autre, sur le même
 * navigateur. Le stockage est un confort, jamais une vérité : une clé
 * absente, illisible ou d'un type inattendu retombe sur la valeur par défaut,
 * et une écriture impossible (quota, navigation privée) est ignorée.
 *
 * Toutes les clés sont préfixées `cw_prefs_` (Compost Waste) pour ne pas
 * entrer en collision avec d'autres applications servies sur la même origine
 * — ni avec les brouillons `compost_brouillon_*` de boxes.ts. */
import { useEffect, useState, type Dispatch, type SetStateAction } from "react"

export const PREFIXE_PREFS = "cw_prefs_"

function lire<T>(cle: string, valeurParDefaut: T): T {
  try {
    const brut = localStorage.getItem(cle)
    if (brut === null) return valeurParDefaut
    const valeur: unknown = JSON.parse(brut)
    // garde-fou : une entrée corrompue ou d'un ancien format ne doit pas
    // casser l'écran — on ne garde que ce qui a le type attendu
    return typeof valeur === typeof valeurParDefaut ? (valeur as T) : valeurParDefaut
  } catch {
    return valeurParDefaut
  }
}

/** Comme `useState`, mais la valeur est relue au montage et réécrite à chaque
 * changement sous la clé `cw_prefs_<nom>`. Les mises à jour fonctionnelles
 * (`set((prev) => …)`) sont acceptées. `nom` est supposé stable. */
export function useLocalStorage<T>(
  nom: string,
  valeurParDefaut: T,
): [T, Dispatch<SetStateAction<T>>] {
  const cle = PREFIXE_PREFS + nom
  const [valeur, setValeur] = useState<T>(() => lire(cle, valeurParDefaut))

  useEffect(() => {
    try {
      localStorage.setItem(cle, JSON.stringify(valeur))
    } catch {
      /* stockage plein ou indisponible : la préférence vaut pour la session */
    }
  }, [cle, valeur])

  return [valeur, setValeur]
}
