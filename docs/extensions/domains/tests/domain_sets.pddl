(define (domain sets-test)

    (:requirements :sets)

    (:types
        item - object
        itemset - (set item)
    )

    (:functions
        (basket) - itemset
    )

    ;; Add an item to the basket if it's not already there
    (:action pick_up
        :parameters (?x - item)
        :precondition (not (member ?x (basket)))
        :effect (add ?x (basket))
    )

    ;; Remove an item from the basket if it's there
    (:action put_down
        :parameters (?x - item)
        :precondition (member ?x (basket))
        :effect (remove ?x (basket))
    )
)
