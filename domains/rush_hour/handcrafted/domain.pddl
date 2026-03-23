(define (domain rush-hour-d1)
    (:requirements :typing :strips)
    (:types position vehicle - object)

    (:predicates
        (ADJACENT ?p1 - position ?p2 - position ?p3 - position)
        (ADJACENT-2 ?p1 - position ?p2 - position)
        (CAR ?v - vehicle)
        (TRUCK ?v - vehicle)
        (QUAD ?v - vehicle)
        (isOccupied ?p - position)
        (containsVehicle ?p - position ?v - vehicle)
        (isWall ?p - position)
    )

    (:functions
        (total-cost) - number
    )

    (:action move-car
        :parameters (?v - vehicle ?p1 - position ?p2 - position ?p3 - position)
        :precondition (and 
            (CAR ?v) 
            (ADJACENT ?p1 ?p2 ?p3)
            (containsVehicle ?p1 ?v)
            (containsVehicle ?p2 ?v)
            (not (isOccupied ?p3))
            (not (isWall ?p3))
        )
        :effect (and
            (not (isOccupied ?p1))
            (not (containsVehicle ?p1 ?v))
            (containsVehicle ?p3 ?v)
            (isOccupied ?p3)
            (increase (total-cost) 1)
        )
    )
    
    (:action move-truck
        :parameters (?v - vehicle ?p1 - position ?p2 - position ?p3 - position ?p4 - position)
        :precondition (and
            (TRUCK ?v) 
            (ADJACENT ?p1 ?p2 ?p3) (ADJACENT ?p2 ?p3 ?p4)
            (containsVehicle ?p1 ?v)
            (containsVehicle ?p2 ?v)
            (containsVehicle ?p3 ?v)
            (not (isOccupied ?p4))
            (not (isWall ?p4))
        )
        :effect (and
            (not (isOccupied ?p1))
            (not (containsVehicle ?p1 ?v))
            (containsVehicle ?p4 ?v)
            (isOccupied ?p4)
            (increase (total-cost) 1)
        )
    )

    (:action move-quad
        :parameters (?v - vehicle ?p1 - position ?p2 - position ?p3 - position ?p4 - position ?p5 - position ?p6 - position)
        :precondition (and
            (QUAD ?v) 
            (ADJACENT ?p1 ?p2 ?p3) (ADJACENT ?p4 ?p5 ?p6)
            (ADJACENT-2 ?p1 ?p4) (ADJACENT-2 ?p2 ?p5) (ADJACENT-2 ?p3 ?p6)
            (containsVehicle ?p1 ?v)
            (containsVehicle ?p2 ?v)
            (containsVehicle ?p4 ?v)
            (containsVehicle ?p5 ?v)
            (not (isOccupied ?p3))
            (not (isOccupied ?p6))
            (not (isWall ?p3))
            (not (isWall ?p6))
        )
        :effect (and
            (not (isOccupied ?p1))
            (not (isOccupied ?p4))
            (not (containsVehicle ?p1 ?v))
            (not (containsVehicle ?p4 ?v))
            (containsVehicle ?p3 ?v)
            (containsVehicle ?p6 ?v)
            (isOccupied ?p3)
            (isOccupied ?p6)
            (increase (total-cost) 1)
        )
    )
)