def write_namd_configuration(conf_file: str, pdb_file: str, args: dict):
    conf = [
        "#############################################################\n",
        "## JOB DESCRIPTION                                         ##\n",
        "#############################################################\n",
        "\n",
        "# Min. and Eq. of KcsA\n",
        "# embedded in POPC membrane, ions and water.\n",
        "# Protein constrained. PME, Constant Pressure.\n",
        "\n",
        "\n",
        "#############################################################\n",
        "## VARIABLES                                               ##\n",
        "#############################################################\n",
        "\n",
        "set namepdb             spike_ace2_equi_pores_lipidsw_ions\n",
        "set inpname      \t../8_PR/pr.8\n",
        "set outname             pr.4\n",
        "\n",
        "\n",
        "set dir_ff              /gpfs/alpine/world-shared/med110/lcasalino/NAMD3_TESTS/SPIKE_ACE_8.5M/SIMULATIONS_SUMMIT/CHARMM36m\n",
        "set dir_pdb             /gpfs/alpine/world-shared/med110/lcasalino/NAMD3_TESTS/SPIKE_ACE_8.5M/SIMULATIONS_SUMMIT\n",
        "\n",
        "\n",
        "set logfreq  \t\t  10080\n",
        "set dcdfreq     \t  %s\n" % args["dcdfreq"],
        "set num_steps_min  \t  %s\n" % args["num_steps_min"],
        "set num_steps_eq   \t  %s;#250000  --> 0.5 ns\n" % args["num_steps_eq"],
        "\n",
        "set temp\t\t310\n",
        "\n",
        "\n",
        "#############################################################\n",
        "## ADJUSTABLE PARAMETERS                                   ##\n",
        "#############################################################\n",
        "\n",
        "structure          ${dir_pdb}/${namepdb}.psf\n",
        "coordinates        %s\n" % pdb_file,
        "outputName         ${outname}\n",
        "\n",
        "# Continuing a job from the restart files\n",
        "if {0} {\n",
        "binCoordinates     $inpname.restart.coor\n",
        "binVelocities      $inpname.restart.vel\n",
        "extendedSystem     $inpname.restart.xsc\n",
        "}\n",
        "\n",
        "firsttimestep      0\n",
        "restartfreq        $dcdfreq     ;# 1000steps = every 2ps\n",
        "dcdfreq            $dcdfreq\n",
        "xstFreq            $dcdfreq\n",
        "outputEnergies     $logfreq\n",
        "outputPressure     $logfreq\n",
        "\n",
        "\n",
        "#############################################################\n",
        "## SIMULATION PARAMETERS                                   ##\n",
        "#############################################################\n",
        "\n",
        "# Input\n",
        "#paraTypeCharmm\t    on\n",
        "#parameters          ../par_all27_prot_lipidNBFIX.prm \n",
        "\n",
        "###############\n",
        "## CHARMM FF ##\n",
        "paraTypeCharmm      on\n",
        "parameters              ${dir_ff}/par_all36m_prot.prm                                           # Protein      \n",
        "parameters              ${dir_ff}/par_all36_lipid.prm                                           # Lipids        \n",
        "parameters              ${dir_ff}/par_all36_carb.prm                        # Carbohydrates\n",
        "parameters              ${dir_ff}/par_all36_cgenff.prm                      # Cgenff\n",
        "parameters              ${dir_ff}/par_all36_na.prm\n",
        "parameters              ${dir_ff}/toppar_water_ions_namd.str                            # Water and ions\n",
        "parameters              ${dir_ff}/toppar_all36_carb_glycopeptide.str\n",
        "parameters              ${dir_ff}/toppar_all36_carb_glycolipid.str\n",
        "parameters              ${dir_ff}/toppar_all36_lipid_cholesterol.str\n",
        "parameters              ${dir_ff}/toppar_all36_lipid_sphingo.str\n",
        "parameters              ${dir_ff}/toppar_all36_palmitoyl_cys.str\n",
        "\n",
        "\n",
        "# NOTE: Do not set the initial velocity temperature if you \n",
        "# have also specified a .vel restart file!\n",
        "#temperature         $temp\n",
        "\n",
        "\n",
        "# Periodic Boundary Conditions\n",
        "# NOTE: Do not set the periodic cell basis if you have also \n",
        "# specified an .xsc restart file!\n",
        "if {1} { \n",
        "cellBasisVector1 325.434997559 0 0\n",
        "cellBasisVector2 0 322.597000122 0\n",
        "cellBasisVector3 0 0 805.089709434\n",
        "cellOrigin -1.33349609375 0.57950592041 200.132011414\n",
        "}\n",
        "wrapWater           on\n",
        "wrapAll             on\n",
        "\n",
        "\n",
        "# Force-Field Parameters\n",
        "exclude             scaled1-4\n",
        "1-4scaling          1.0\n",
        "cutoff              12.\n",
        "switching           on\n",
        "switchdist          10.\n",
        "pairlistdist        13.5\n",
        "\n",
        "\n",
        "# Integrator Parameters\n",
        "timestep            2.0  ;# 2fs/step\n",
        "rigidBonds          all  ;# needed for 2fs steps\n",
        "nonbondedFreq       1\n",
        "fullElectFrequency  %s  \n" % args.get("fullElectFrequency", "3"),
        "stepspercycle       %s\n" % args["stepspercycle"],
        "\n",
        "\n",
        "################################################\n",
        "## PME (for full-system periodic electrostatics)\n",
        "PME\t          on\t\t\t\t#\n",
        "PMEInterpOrder    8\t\t\t\t#\n",
        "PMEGridSpacing    2\t\t\t#\n",
        "#PMEGridSizeX      640\t\t\t\t#\n",
        "#PMEGridSizeY      640\t\t\t\t#\n",
        "#PMEGridSizeZ      672\t\t\t\t#\n",
        "\n",
        "\n",
        "# Constant Temperature Control\n",
        "langevin            on    ;# do langevin dynamics\n",
        "langevinDamping     1     ;# damping coefficient (gamma) of 5/ps\n",
        "langevinTemp        $temp\n",
        "\n",
        "# Constant Pressure Control (variable volume)\n",
        "if {1} {\n",
        "useGroupPressure      yes ;# needed for 2fs steps\n",
        "useFlexibleCell       yes  ;# no for water box, yes for membrane\n",
        "useConstantArea       yes  ;# no for water box, yes for membrane\n",
        "\n",
        "langevinPiston        on\n",
        "langevinPistonTarget  1.01325 ;#  in bar -> 1 atm\n",
        "langevinPistonPeriod  200.\n",
        "langevinPistonDecay   50.\n",
        "langevinPistonTemp    $temp\n",
        "}\n",
        "\n",
        "\n",
        "# Fixed Atoms Constraint (set PDB beta-column to 1)\n",
        "if {0} {\n",
        "fixedAtoms          on\n",
        "fixedAtomsFile      nottails.fix.pdb\n",
        "fixedAtomsCol       B\n",
        "fixedAtomsForces    on\n",
        "}\n",
        "\n",
        "#############################################################\n",
        "## EXTRA PARAMETERS                                        ##\n",
        "#############################################################\n",
        "\n",
        "# Put here any custom parameters that are specific to \n",
        "# this job (e.g., SMD, TclForces, etc...)\n",
        "\n",
        "#constraints \ton\n",
        "#consexp\t\t2\n",
        "#consref\t\t${dir_pdb}/${namepdb}.pdb\n",
        "#conskfile\t${dir_pdb}/${namepdb}.proteinglycanrestrained.cnst\n",
        "#conskcol\tB\n",
        "%smargin\t\t3\n" % ("" if "margin" in args else "#"),
        "#\n",
        "#tclforces\t\t\ton\n",
        "#set waterCheckFreq              100\n",
        "#set lipidCheckFreq              100\n",
        "#set allatompdb                  ${dir_pdb}/${namepdb}.pdb\n",
        "#tclForcesScript                 ${dir_pdb}/keep_water_out_lc_2m_norm.tcl\n",
        "\n",
        "#eFieldOn yes\n",
        "#eField 0 0 -0.155\n",
        "\n",
        "\n",
        "#############################################################\n",
        "## EXECUTION SCRIPT                                        ##\n",
        "#############################################################\n",
        "\n",
        "# Minimization\n",
        "if {0} {\n",
        "minimize            $num_steps_min\n",
        "reinitvels          $temp\n",
        "}\n",
        "temperature		$temp\n",
        "run		    $num_steps_eq\n",
    ]

    with open(conf_file, "w") as f:
        f.writelines(conf)
