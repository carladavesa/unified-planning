(define (domain dump-trucks)

    (:requirements :typing :arrays :bounded-integers)

    (:types
        location    - object
        truck       - object
        package     - object
        pckg_set    - (set package)
        loc_set     - (set location)
    )

    (:predicates
        (truck_at ?t - truck ?l - location)
    )

    (:functions
        (package_at ?l - location)  - pckg_set
        (package_in ?t - truck)     - pckg_set
        (connects ?l - location)    - loc_set
    )

    (:action move_truck
        :parameters (?to ?from - location ?t - truck)
        :precondition (and
            (member ?to (connects ?from))
            (truck_at ?t ?from)
        )
        :effect (and
            (not (truck_at ?t ?from))
            (truck_at ?t ?to)
        )
    )

    (:action load_truck
        :parameters (?p - package ?t - truck ?l - location)
        :precondition (and
            (truck_at ?t ?l)
            (member ?p (package_at ?l))
            (< (cardinality (package_in ?t)) 5) ; !!!!
        )
        :effect (and
            (remove ?p (package_at ?l))
            (add ?p (package_in ?t))
        )
    )

    (:action unload_truck
        :parameters (?p - package ?t - truck ?l - location)
        :precondition (and
            (truck_at ?t ?l)
        )
        :effect (and
            (assign (package_at ?l) (union (package_at ?l) (package_in ?t)))
            (assign (package_in ?t) (difference (package_in ?t) (package_in ?t)))
        )
    )
)
