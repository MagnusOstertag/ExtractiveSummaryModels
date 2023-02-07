Each json file has the following format:
- "dialogue": This is the input dialogue
- "relevant_dialogue": This is a list of sentences from the dialogue that are considered relevant
- "finegrained_relevant_dialogue": This is a list of sequences of relevant tokens extracted from the dialogue; it is a finegrained version of the "relevant_dialogue" field

{
        "dialogue": "# Eric: MACHINE! # Rob: That's so gr8! # Eric: I know! And shows how Americans see Russian ;) # Rob: And it's really funny! # Eric: I know! I especially like the train part! # Rob: Hahaha! No one talks to the machine like that! # Eric: Is this his only stand-up? # Rob: Idk. I'll check. # Eric: Sure. # Rob: Turns out no! There are some of his stand-ups on youtube. # Eric: Gr8! I'll watch them now! # Rob: Me too! # Eric: MACHINE! # Rob: MACHINE! # Eric: TTYL? # Rob: Sure :)\n",
        "relevant_dialogue": [
            "# Eric: Is this his only stand-up?",
            "# Rob: Turns out no! There are some of his stand-ups on youtube.",
            "# Eric: Gr8! I'll watch them now!"
        ],
        "finegrained_relevant_dialogue": [
            "his only stand-up",
            "are some of his stand-ups on youtube",
            "'ll watch them now"
        ]
    }