(define (problem instance13921)
(:domain ext-plant-watering)
(:objects
	plant1 - plant
	plant2 - plant
	plant3 - plant
	plant4 - plant
	plant5 - plant
	plant6 - plant
	plant7 - plant
	plant8 - plant
	plant9 - plant
	tap1 - tap
	agent1 - agent
	agent2 - agent
)
(:init
	(= (maxx) 13)
	(= (minx) 1)
	(= (maxy) 13)
	(= (miny) 1)
	(= (totalpoured) 0)
	(= (totalloaded) 0)
	(= (waterreserve) 38)
	(= (carrying agent1) 0)
	(= (maxcarry agent1) 5)
	(= (carrying agent2) 0)
	(= (maxcarry agent2) 5)
	(= (poured plant1) 0)
	(= (poured plant2) 0)
	(= (poured plant3) 0)
	(= (poured plant4) 0)
	(= (poured plant5) 0)
	(= (poured plant6) 0)
	(= (poured plant7) 0)
	(= (poured plant8) 0)
	(= (poured plant9) 0)
	(= (x plant1) 5)
	(= (y plant1) 11)
	(= (x plant2) 8)
	(= (y plant2) 12)
	(= (x plant3) 5)
	(= (y plant3) 12)
	(= (x plant4) 8)
	(= (y plant4) 3)
	(= (x plant5) 9)
	(= (y plant5) 11)
	(= (x plant6) 3)
	(= (y plant6) 2)
	(= (x plant7) 5)
	(= (y plant7) 9)
	(= (x plant8) 7)
	(= (y plant8) 11)
	(= (x plant9) 12)
	(= (y plant9) 13)
	(= (x tap1) 9)
	(= (y tap1) 4)
	(= (x agent1) 8)
	(= (y agent1) 4)
	(= (x agent2) 10)
	(= (y agent2) 4)
)
(:goal
(and
	(= (poured plant1) 6)
	(= (poured plant2) 6)
	(= (poured plant3) 6)
	(= (poured plant4) 3)
	(= (poured plant5) 3)
	(= (poured plant6) 2)
	(= (poured plant7) 2)
	(= (poured plant8) 2)
	(= (poured plant9) 5)
	(= (totalpoured) (totalloaded))
)))
