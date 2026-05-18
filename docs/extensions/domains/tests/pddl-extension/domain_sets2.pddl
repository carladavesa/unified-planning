(define (domain sets-test2)

    (:requirements :sets)

    (:types
        item - object
        itemset - (set item)
    )

    (:functions
        (bag1) - itemset
        (bag2) - itemset
    )

    ;; Merge bag2 into bag1 when the two bags are disjoint → bag1 := bag1 ∪ bag2
    (:action merge
        :parameters ()
        :precondition (disjoint (bag1) (bag2))
        :effect (assign (bag1) (union (bag1) (bag2)))
    )

    ;; Retain only the items common to both bags → bag1 := bag1 ∩ bag2
    (:action keep_common
        :parameters ()
        :precondition (and)
        :effect (assign (bag1) (intersect (bag1) (bag2)))
    )

    ;; When bag1 ⊆ bag2, flip bag1 to the complement: bag1 := bag2 \ bag1
    (:action take_complement
        :parameters ()
        :precondition (subset (bag1) (bag2))
        :effect (assign (bag1) (difference (bag2) (bag1)))
    )

    ;; Add a single item to bag1 if it is not already there
    (:action add_item
        :parameters (?x - item)
        :precondition (not (member ?x (bag1)))
        :effect (add ?x (bag1))
    )
)
