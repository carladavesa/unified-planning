(define (problem instance15721)
(:domain ext-plant-watering)
(:objects
	plant1 - plant
	plant2 - plant
	plant3 - plant
	plant4 - plant
	plant5 - plant
	plant6 - plant
	plant7 - plant
	tap1 - tap
	agent1 - agent
	agent2 - agent
)
(:init
	(= (maxx) 15)
	(= (minx) 1)
	(= (maxy) 15)
	(= (miny) 1)
	(= (totalpoured) 0)
	(= (totalloaded) 0)
	(= (waterreserve) 51)
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
	(= (x plant1) 8)
	(= (y plant1) 12)
	(= (x plant2) 7)
	(= (y plant2) 1)
	(= (x plant3) 8)
	(= (y plant3) 8)
	(= (x plant4) 12)
	(= (y plant4) 13)
	(= (x plant5) 1)
	(= (y plant5) 5)
	(= (x plant6) 2)
	(= (y plant6) 13)
	(= (x plant7) 1)
	(= (y plant7) 7)
	(= (x tap1) 8)
	(= (y tap1) 4)
	(= (x agent1) 1)
	(= (y agent1) 4)
	(= (x agent2) 12)
	(= (y agent2) 15)
)
(:goal
(and
	(= (poured plant1) 8)
	(= (poured plant2) 7)
	(= (poured plant3) 9)
	(= (poured plant4) 7)
	(= (poured plant5) 4)
	(= (poured plant6) 5)
	(= (poured plant7) 7)
	(= (totalpoured) (totalloaded))
)))
