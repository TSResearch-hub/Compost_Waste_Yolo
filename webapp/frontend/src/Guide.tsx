/** Guide d'annotation intégré — consultable sans quitter l'écran (touche G).
 * Le contenu reprend les règles validées par l'équipe : à ne pas modifier
 * sans elles. L'ordre des classes suit data.yaml (index = class_id). */
import { couleurClasse } from "./boxes"

interface LigneClasse {
  classe: string
  annoter: string
  pasIci: string
}

// index = class_id (ordre data.yaml : Plastique, Metal, Carton, Aluminium,
// Ceramique, Verre, Composite, Eponge)
const TABLE_CLASSES: LigneClasse[] = [
  {
    classe: "Plastique",
    annoter: "Bouteille, sachet, film, paille, pot de yaourt, couverts plastique",
    pasIci: "Barquette aluminium, éponge",
  },
  {
    classe: "Métal",
    annoter: "Boîte de conserve, couvercle, couverts, trombone, ferraille",
    pasIci: "Canette de boisson, papier aluminium, pile",
  },
  {
    classe: "Carton",
    annoter: "Morceau de carton, boîte aplatie, étiquette, emballage papier",
    pasIci: "Carton décomposé, indissociable du compost",
  },
  {
    classe: "Aluminium",
    annoter: "Papier alu, barquette, capsule, canette de boisson",
    pasIci: "Boîte de conserve en acier",
  },
  {
    classe: "Céramique",
    annoter: "Assiette, bol, vaisselle cassée, carrelage, pot en terre",
    pasIci: "Tesson de verre",
  },
  {
    classe: "Verre",
    annoter: "Bouteille, bocal, tesson, verre à boire",
    pasIci: "Plastique transparent",
  },
  {
    classe: "Composite",
    annoter: "Piles et batteries uniquement",
    pasIci: "Tout autre objet multi-matériaux",
  },
  {
    classe: "Éponge",
    annoter: "Éponge de nettoyage, grattoir",
    pasIci: "Mousse d'emballage",
  },
]

export default function Guide({ onFermer }: { onFermer: () => void }) {
  return (
    <div className="guide-voile" onClick={onFermer}>
      <div className="guide-panneau" onClick={(e) => e.stopPropagation()}>
        <div className="guide-entete">
          <h2>Guide d'annotation</h2>
          <button className="btn btn-petit" onClick={onFermer}>
            Fermer (Échap)
          </button>
        </div>

        <table className="guide-table">
          <thead>
            <tr>
              <th>Classe</th>
              <th>À annoter</th>
              <th>À ne pas mettre ici</th>
            </tr>
          </thead>
          <tbody>
            {TABLE_CLASSES.map((l, idx) => (
              <tr key={l.classe}>
                <td className="guide-classe">
                  <span className="pastille" style={{ background: couleurClasse(idx) }} />
                  {l.classe}
                  <span className="btn-classe-chiffre"> {idx + 1}</span>
                </td>
                <td>{l.annoter}</td>
                <td className="texte-doux">{l.pasIci}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <h3>Règles</h3>
        <ul className="guide-regles">
          <li>
            On annote le <strong>matériau</strong>, pas l'objet.
          </li>
          <li>Le compost est le fond, il n'est jamais annoté.</li>
          <li>Une boîte par objet, au plus juste, sans marge.</li>
          <li>Objet partiellement enfoui : n'englober que la partie visible.</li>
          <li>Tesson brisé en plusieurs morceaux : une boîte par morceau.</li>
          <li>
            <strong>En cas de doute, ne pas deviner</strong> — laisser sans boîte et soumettre le
            cas à l'équipe.
          </li>
          <li>Une image sans intrus est valide et utile : elle se valide sans aucune boîte.</li>
          <li>
            <strong>Verre / Céramique</strong> : le verre est transparent ou translucide, la
            céramique est opaque et mate sur la tranche.
          </li>
          <li>
            <strong>Aluminium / Métal</strong> : canette, papier alu, barquette, capsule →
            Aluminium ; boîte de conserve, couvercle, couverts → Métal.
          </li>
          <li>
            <strong>Composite</strong> ne contient que les piles et batteries, malgré son nom.
          </li>
          <li>
            Le modèle ne propose <strong>jamais</strong> de boîte Éponge : ces objets se tracent
            entièrement à la main.
          </li>
        </ul>
      </div>
    </div>
  )
}
