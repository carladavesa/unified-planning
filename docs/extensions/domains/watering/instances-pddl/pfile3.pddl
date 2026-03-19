(define (problem instance10721)
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
	(= (maxx) 10)
	(= (minx) 1)
	(= (maxy) 10)
	(= (miny) 1)
	(= (totalpoured) 0)
	(= (totalloaded) 0)
	(= (waterreserve) 53)
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
	(= (x plant1) 6)
	(= (y plant1) 6)
	(= (x plant2) 3)
	(= (y plant2) 10)
	(= (x plant3) 8)
	(= (y plant3) 8)
	(= (x plant4) 8)
	(= (y plant4) 1)
	(= (x plant5) 2)
	(= (y plant5) 5)
	(= (x plant6) 2)
	(= (y plant6) 6)
	(= (x plant7) 8)
	(= (y plant7) 10)
	(= (x tap1) 7)
	(= (y tap1) 10)
	(= (x agent1) 3)
	(= (y agent1) 8)
	(= (x agent2) 1)
	(= (y agent2) 9)
)
(:goal
(and
	(= (poured plant1) 10)
	(= (poured plant2) 7)
	(= (poured plant3) 3)
	(= (poured plant4) 9)
	(= (poured plant5) 9)
	(= (poured plant6) 1)
	(= (poured plant7) 10)
	(= (totalpoured) (totalloaded))
)))
