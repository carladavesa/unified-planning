(define (problem labyrinth-size-4-rotations-20-seed-5)
(:domain labyrinth)
(:objects
	pos0 pos1 pos2 pos3  - gridpos
	card0 card1 card2 card3 card4 card5 card6 card7 card8 card9 card10 card11 card12 card13 card14 card15  - card
)
(:init
	(MAX-POS pos3)
	(MIN-POS pos0)

	(NEXT pos1 pos0)
	(NEXT pos2 pos1)
	(NEXT pos3 pos2)

	(card-at card0 pos0 pos0)
	(card-at card10 pos1 pos0)
	(card-at card9 pos2 pos0)
	(card-at card3 pos3 pos0)
	(card-at card8 pos0 pos1)
	(card-at card13 pos1 pos1)
	(card-at card7 pos2 pos1)
	(card-at card11 pos3 pos1)
	(card-at card4 pos0 pos2)
	(card-at card1 pos1 pos2)
	(card-at card2 pos2 pos2)
	(card-at card5 pos3 pos2)
	(card-at card12 pos0 pos3)
	(card-at card6 pos1 pos3)
	(card-at card14 pos2 pos3)
	(card-at card15 pos3 pos3)

	(BLOCKED card0 N)
	(BLOCKED card0 E)

	(BLOCKED card10 E)

	(BLOCKED card9 N)
	(BLOCKED card9 S)


	(BLOCKED card8 S)
	(BLOCKED card8 W)

	(BLOCKED card13 W)

	(BLOCKED card7 E)
	(BLOCKED card7 W)

	(BLOCKED card11 E)
	(BLOCKED card11 W)

	(BLOCKED card4 E)
	(BLOCKED card4 W)

	(BLOCKED card1 N)
	(BLOCKED card1 W)

	(BLOCKED card2 N)

	(BLOCKED card5 S)
	(BLOCKED card5 W)

	(BLOCKED card12 E)
	(BLOCKED card12 S)

	(BLOCKED card6 N)
	(BLOCKED card6 E)

	(BLOCKED card14 S)
	(BLOCKED card14 W)

	(BLOCKED card15 E)
	(BLOCKED card15 W)


	(robot-at card0)

	(= (total-cost) 0)
	(= (move-robot-cost) 1)
	(= (move-card) 1)
)
(:goal
	(and
		(left)
	)
)
	(:metric minimize (total-cost))
)
