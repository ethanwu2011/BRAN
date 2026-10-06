"""Final display numbers for the BRAN submission, ordered by first citation.

Builders import this module for figure titles and cross-references, so a renumbering
happens in one place. Internal builder names stay unchanged.
"""

MAIN = {
    "fig1": "Fig. 1",
    "fig2": "Fig. 2",
    "fig3": "Fig. 3",
    "fig4": "Fig. 4",
    "fig5": "Fig. 5",
    "fig6": "Fig. 6",
}

EXTENDED = {
    "ed1": "Extended Data Fig. 1",      # one state, many uses, and the predecessor test
    "ed2": "Extended Data Fig. 2",      # participant flow and Table 1
    "ed3": "Extended Data Fig. 3",      # representation comparison, fixed readout and per condition
    "ed3b": "Extended Data Fig. 4",     # equal tuning, readout sensitivity, eye diagnostic, R8 retrain
    "ed4": "Extended Data Fig. 5",      # blood count completion details
    "fig6A": "Extended Data Fig. 6",    # MIMIC-IV recovery deficit
    "ed8": "Extended Data Fig. 7",      # subgroup attempts
    "ed_table1": "Extended Data Table 1",   # cohort exposure and test history
    "ed_table2": "Extended Data Table 2",   # Figure 4 operating points and referral use
    "ed_table3": "Extended Data Table 3",   # Figure 6 detail, built with the Figure 6 work
}


def label(internal_id: str) -> str:
    if internal_id in MAIN:
        return MAIN[internal_id]
    return EXTENDED[internal_id]
