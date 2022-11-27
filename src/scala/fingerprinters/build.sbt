val scala3Version = "3.2.1"

lazy val root = project
  .in(file("."))
  .settings(
    name := "fingerprinters",
    version := "0.1.0-SNAPSHOT",

    scalaVersion := scala3Version,

    libraryDependencies ++= Seq(
      "org.openscience.cdk" % "cdk-bundle" % "2.8"
    ),

    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", xs @ _*) => MergeStrategy.discard
      case x => MergeStrategy.first
    },

    assembly / assemblyJarName := "fingerprinters.jar"
  )

