import org.openscience.cdk.fingerprint.PubchemFingerprinter

import org.openscience.cdk.fingerprint.{CircularFingerprinter, MACCSFingerprinter, PubchemFingerprinter}
import org.openscience.cdk.silent.SilentChemObjectBuilder
import org.openscience.cdk.smiles.SmilesParser
import scala.util.Try

package co.insilica.vectorizer:

  object ChemicalVectorizer{

    object fingerprints{
      val pubchem2d = "pubchem2d"
      val maccs = "maccs"
      val circular = "circular"
    }

    def fingerprint(smiles:String,fingerprinter:String) : Seq[Array[Int]] = {
      val builder = SilentChemObjectBuilder.getInstance()
      val sp = new SmilesParser(builder)
      val fp =  fingerprinter match {
        case fingerprints.pubchem2d => new PubchemFingerprinter(builder)
        case fingerprints.maccs => new MACCSFingerprinter()
        case fingerprints.circular => new CircularFingerprinter()
      }

      val mol = sp.parseSmiles(smiles) 
      val bfp = fp.getBitFingerprint(mol)

      Seq(bfp.getSetbits.toArray)
    }

    def hi(): Int = 5
  }