[Input]

**ssf** *state simple flat*
773 features:
12x8x8=768 chess board flattened (12 piece types - 6 per color)
4 casteling rights
1 active color

[Output]

**asf** *action simple flat*
4096=64x8x8 features (8x8 piece choice - 64 move choice)
8x7 queen moves (8 directions - 7 dist)
8 knight moves
no underpromotions


