# Rights status

The project owner confirmed ownership of the BRAN code on 1 October 2026 and explicitly approved the MIT license. The license is in `LICENSE`. Model weights and clinical data are outside the code-release scope.

Third-party packages remain external dependencies and retain their own distribution terms. The code license must not be read as a license to clinical datasets, retinal models or separately obtained research artifacts.

## Source Data and third-party assets

The Source Data in `paper/source_data` are aggregates derived from datasets held under the PhysioNet and AI-READI agreements. They are released because every value passes the disclosure rules in `paper/source_data/README.md`, and the same workbooks go to the journal with the paper. PLACEHOLDER for the author's decision on their licence in this repository, MIT with the code or the licence the journal applies to Source Data.

The four icons in `paper/source_data/fig1_source_data.json` (`draw.icons`, named retina, tube, cuff and person_5) were drawn with BioRender. They are third-party artwork, not covered by the MIT license in `LICENSE`, and the public rebuild of Fig. 1 needs them. PLACEHOLDER for written confirmation that the BioRender plan of the author allows these icon files to be redistributed in a public repository. The repository must not be pushed or made public before that confirmation. If redistribution is not allowed, the branch history must be rewritten to remove `draw.icons` and Fig. 1 redrawn with openly licensed or self-drawn symbols before the first push.

Fonts are not distributed. The drawing code finds them on the reader's system.
