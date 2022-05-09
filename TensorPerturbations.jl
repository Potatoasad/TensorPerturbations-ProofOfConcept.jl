### A Pluto.jl notebook ###
# v0.18.0

using Markdown
using InteractiveUtils

# ╔═╡ 4c467392-ccde-11ec-15c0-4f532000315f
begin
	using LaTeXStrings
	using Symbolics
	md"# Tensor Perturbations"
end

# ╔═╡ 977b2170-886d-40d8-ada2-29385c8c8bde
begin
import SymbolicUtils: Sym, FnType, Term, Add, Mul, Pow, similarterm
using SymbolicUtils

# Create a new number type simply for dispatch purposes.
# Everytime you define a tensor manipulation system, you would define a new number type

struct NewTensor <: Number end

# Create a TensorDisplay module
abstract type AbstractTensorDisplay end

"""
This contains all the info to display a tensor 
"""
struct TensorDisplay{T <: AbstractString} <: AbstractTensorDisplay
	name::T
	sub::T
	sup::T
end

TensorDisplay(name::AbstractString; sub="", super="") = TensorDisplay(name,sub,super)
append_sub(x::TensorDisplay,a::AbstractString) = TensorDisplay(x.name,x.sub*a,x.sup)
append_sub(a::AbstractString) = (x -> append_sub(x,a))
append_sup(x::TensorDisplay,a::AbstractString) = TensorDisplay(x.name,x.sub,x.sup*a)
append_sup(a::AbstractString) = (x -> append_sup(x,a))
append_name(x::TensorDisplay,a::AbstractString) = TensorDisplay(x.name*a,x.sub,x.sup)
append_name(a::AbstractString) = (x -> append_name(x,a))

"""
Function to extract the latex representation of the TensorDisplay as a string. 
ex: str(TensorDisplay("T","a","\textrm{int}")) = "T_{a}^{\textrm{int}}"
"""
function str(x::TensorDisplay)
	sub = x.sub == "" ? "" : "_{$(x.sub)}"
	sup = x.sup == "" ? "" : "^{$(x.sup)}"
	"$(x.name)$(sup)$(sub)" 
end
str(x::AbstractString) = x
subscript(x::TensorDisplay) = x.sub
superscript(x::TensorDisplay) = x.sup
name(x::TensorDisplay) = x.name

# Define the Display properties of the NewTensor type
TensorDisplay(x::Term{NewTensor}) = getmetadata(operation(x),Type{TensorDisplay})
subscript(x::Term{NewTensor}) = TensorDisplay(x).sub
superscript(x::Term{NewTensor}) = TensorDisplay(x).sup
name(x::Term{NewTensor}) = TensorDisplay(x).name

# utility function
intersperse(a::Vector; token = ",") = [(i % 2) == 0 ? token : a[i÷2 + 1] for i ∈ 1:(2*length(a)-1)]

md"> Created a new Tensor type and defined it's displays"
end

# ╔═╡ 9bd50b69-1f6d-4a5a-855e-15d70657742f
begin
md"""
What are the main components of this package and how do they work together?

### Operators
You can define operator types. Examples include something like $T_{[\textrm{int}]}^{(1,2)}[g^0,h^0]$ or $\hat{H}|\psi>$. The first main thing an operator can do, is display itself. 

Every operator has a property stored in a struct called `TensorDisplay`. This keeps the latex display of the operator head, the subscript and the superscript. One can also override the `_toexpr` function to get the correct display for latexify. 

In and of itself, the operator is nothing but a pretty display. The real power of the operator is that it is a function that accepts things in slots. Each operator has a particular structure in it's slots. The two structures we have implemented are _Multilinearity_ and _SlotSymmetry_.

"""
end

# ╔═╡ eb6f19bd-f750-478b-967e-39710e4f42c2
begin

abstract type AbstractSlot end
struct Slot <: AbstractSlot end

abstract type AbstractSlotStructure end

abstract type MultilinearSlotStructure <: AbstractSlotStructure end
abstract type SymmetrySlotStructure <: AbstractSlotStructure end

struct Multilinear <: MultilinearSlotStructure
	indices::Vector{Int64}
end

struct NotScalar{T <: MultilinearSlotStructure} end 
	
struct TotalSymmetry <: SymmetrySlotStructure
	indices::Vector{Vector{Int64}}
end
	
struct Slots{N}
	slot_structures::Dict{DataType, AbstractSlotStructure}
end
Slots{N}(x::AbstractSlotStructure...) where {N} = Slots{N}(Dict([Type{typeof(x1)} => x1 for x1 in x]))
merge_together(x::Slots{N}, y::Slots{M}) where {N,M} = Slots{M}(merge(x.slot_structures,y.slot_structures))
remove_structure!(x::Slots, v::DataType) = delete!(x.slot_structures, v)

slot_structure(x::Slots) = x.slot_structures
	
is_multilinear(x::Slots) = Type{Multilinear} ∈ keys(x.slot_structures)
is_multilinear(x::Term{NewTensor}) = is_multilinear(Slots(x))
is_symmetric(x::Slots) = Type{TotalSymmetry} ∈ keys(x.slot_structures)
is_symmetric(x::Term{NewTensor}) = is_symmetric(Slots(x))
	
multilinear_indices(x::Slots) = x.slot_structures[Type{Multilinear}].indices
symmetric_indices(x::Slots) = x.slot_structures[Type{TotalSymmetry}].indices
	
is_fully_multilinear(x::Slots{N}) where {N} = is_multilinear(x) && (Set(multilinear_indices(x)) == Set(1:N))
is_fully_multilinear(x::Term{NewTensor}) = is_fully_multilinear(Slots(x))

Slots(x::Term{NewTensor}) = getmetadata(operation(x),Type{Slots})
	
number_of_slots(x::Slots{N}) where {N} = N
number_of_slots(x::Term{NewTensor}) = number_of_slots(Slots(x))
	
md"> Created a new Slot type"
end

# ╔═╡ 004ab9ee-4562-421a-ad7c-eaa91c8f96db
begin
import Symbolics: _toexpr

variable(s::AbstractString) = Sym{Number}(Symbol(s))
function variable(s::AbstractString, x::Dict{DataType,S}) where S
	var = variable(s)
	for (t,p) ∈ x
		var = setmetadata(var, t, p)
	end
	var
end
variable(s::Union{AbstractString,TensorDisplay}, x::Pair{DataType, S}) where {K,S} = setmetadata(variable(str(s)), x.first, x.second)

operator(s::TensorDisplay, K::Slots{N}) where {N} = setmetadata(setmetadata(Sym{FnType{NTuple{N,Number},NewTensor}}(Symbol(str(s))), Type{TensorDisplay}, s),Type{Slots},K)
operator(s::TensorDisplay, K::Slots, x::Pair{DataType, S}) where {S} = setmetadata(operator(s),x.first, x.second)

function operator(s::TensorDisplay, sl::Slots, x::Union{Dict{DataType,S},Base.ImmutableDict{DataType,S}}) where {S}
	op = operator(s,sl)
	for (t,p) ∈ x
		op = setmetadata(op, t, p)
	end
	op
end

# Overload the _toexpr function in Symbolics to have this custom tensor term
# display as you want it
function _toexpr(x::Term{NewTensor})
	is_linear = is_fully_multilinear(x)
	b = is_linear ?  ["[","]"] : ["(",")"]
	Expr(:latexifymerge, operation(x),b[1], intersperse(arguments(x))..., b[2])
end


md"> Functions to create new variables and operators"
end

# ╔═╡ 1f229cd6-8ab9-4c37-a598-b9a3b2d65c94
begin
import SymbolicUtils: Postwalk, Fixpoint, Prewalk, PassThrough

function partial_reorder(main_list::Vector, symmetry::Vector; by=identity)
	reorder = symmetry[sortperm(main_list[symmetry]; by=by)]
	result = collect(1:length(main_list))
	result2 = 1:length(main_list)
	for i ∈ 1:length(symmetry)
		if symmetry[i] != reorder[i]
			result[symmetry[i]] = result2[reorder[i]]
		end
	end
	main_list[result]
end

function partial_reorder(main_list::Vector, symmetry_list::Vector{Vector{T}}; by=identity) where {T}
	p = main_list
	for symmetry in symmetry_list
		p = partial_reorder(p, symmetry; by=by)
	end
	p
end


already_partially_ordered(x; by=hash) = true
function already_partially_ordered(x::Term{NewTensor}; by=hash)
	if !is_symmetric(x)
		return true
	end
	args = arguments(x)
	sym_inds = symmetric_indices(Slots(x))
	ordered_args = sort(args; by=hash)
	for i in 1:length(args)
		if by(args[i]) != by(ordered_args[i])
			return false
		end
	end
	return true
end

canonicalize_term(x; by=hash) = x
function canonicalize_term(x::Term{NewTensor}; by=hash)
	args = partial_reorder(arguments(x),symmetric_indices(Slots(x));by=by)
	similarterm(x,operation(x),args)
end

r = @rule ~x::(z -> !already_partially_ordered(z)) => canonicalize_term(x)
canonicalize(x) = simplify(x, Prewalk(PassThrough(r)))


md"> Symmetry canonicalizations"
end

# ╔═╡ 893d798e-7376-43bc-99b1-2e65f45f1c18
begin
param(s::Type{K}) where {K <: MultilinearSlotStructure} = K 
	
is_scalar(x, s::Type{<:MultilinearSlotStructure}) = true
is_scalar(x::Number, s::Type{<:MultilinearSlotStructure}) = true
is_scalar(x::Sym, s::Type{<:MultilinearSlotStructure}) = !hasmetadata(x, Type{NotScalar{param(s)}})

argument_should_expand(x::Number, s::Type{<:MultilinearSlotStructure}) = (x == one(x)) ? false : true
argument_should_expand(x::Sym, s::Type{<:MultilinearSlotStructure}) = true
argument_should_expand(x::Add, s::Type{<:MultilinearSlotStructure}) = true
argument_should_expand(x::Mul, s::Type{<:MultilinearSlotStructure}) = any((x -> is_scalar(x,s)).(arguments(x)))
argument_should_expand(s::Type{<:MultilinearSlotStructure}) = (x -> argument_should_expand(x,s))

should_expand(x,s::Type{<:MultilinearSlotStructure}) = false
should_expand(x::Term{NewTensor},s::Type{<:MultilinearSlotStructure}) = is_multilinear(x) && any(argument_should_expand(s).(arguments(x)))

function expand_additions(x,s::Type{<:MultilinearSlotStructure})
	is_scalar(x,s) ? (x,1) : (1,x)
end
function expand_additions(x::Mul, s::Type{<:MultilinearSlotStructure})
	scalars = []; non_scalars = [];
	for arg in arguments(x)
		is_scalar(arg,s) ? push!(scalars, arg) : push!(non_scalars,arg)
	end
	if length(scalars) == 0
		scalars = [1]
	end
	if length(non_scalars) == 0
		non_scalars = [1]
	end
	Tuple([operation(x)(scalars...),operation(x)(non_scalars...)])
end
make_list_if_not(x) = [x]
make_list_if_not(x::Vector) = x
expand_additions(x::Add, s::Type{<:MultilinearSlotStructure}) = (x -> expand_additions(x,s)).(arguments(x))
expand_additions(s::Type{<:MultilinearSlotStructure}) = (x -> expand_additions(x,s))

function expand_linear_term(x::Term{NewTensor}, s::Type{<:MultilinearSlotStructure})
	F = operation(x)
	arg_tuples = expand_additions(s).(expand.(arguments(x)))
	arg_tuples = make_list_if_not.(expand_additions(s).(expand.(arguments(x))))
	newstuff = Iterators.product(arg_tuples...) |> collect |> vec
	sum([prod([p[1] for p ∈ a])*F([p[2] for p ∈ a]...) for a ∈ newstuff])
end
expand_linear_term(s::Type{<:MultilinearSlotStructure}) = (x -> expand_linear_term(x,s))

r2(s::Type{<:MultilinearSlotStructure}) = @rule ~x::(z -> should_expand(z,s)) => expand_linear_term(x,s)

expand_linear(x,s::Type{<:MultilinearSlotStructure}) = simplify(x,Prewalk(PassThrough(r2(s))))

expand_linear(s::Type{<:MultilinearSlotStructure}) = (x -> expand_linear(x,s))
	
md"> Adding Linearity"
end

# ╔═╡ 223ae6fa-bbae-42ba-a4ea-6dc2814aa521
begin

abstract type PerturbationOrder end

abstract type AbstractPerturbationParameters{T} end

const APT{T} = AbstractPerturbationParameters{T} where {T}

struct PerturbationParameters{T} <: AbstractPerturbationParameters{T}
	type_list::Vector{DataType}
	params::Vector{T}
end

PerturbationParameters(x::Dict{DataType, T}) where {T} = PerturbationParameters(collect(keys(x)),collect(values(x)))


pert(x, p::DataType)::Integer = 0
pert(x::Sym, p::DataType)::Integer = hasmetadata(x, p) ? getmetadata(x, p) : 0
pert(x::Mul, p::DataType)::Integer = sum(pert(a, p) for a ∈ arguments(x))
pert(x::Pow, p::DataType)::Integer = pert(x.base, p)*x.exp
pert(x, ps::Vector{DataType}) = [pert(x, p) for p ∈ ps]
pert(x, Ξ::APT) = [pert(x, p) for p ∈ Ξ.type_list]
pert(Ξ::APT) = (x -> pert(x,Ξ))
pert(x, pss::Vector{Sym}) = pert(x, [collect(keys(p.metadata)) for p in pss])

filter(x::Add, p::DataType, i::Integer) = similarterm(x, operation(x), [a for a in arguments(x) if pert(a,p) == i])

filter(x::Add, ps::Vector{DataType}, i_vec::Vector{<:Integer}) where {T <: PerturbationOrder} = similarterm(x, operation(x), [a for a in arguments(x) if all((pert(a,p) == i) for (i,p) ∈ zip(i_vec, ps))])

pert_like(x::Union{Tuple,Vector},Ξ::APT) = prod((Ξ.params[i]^x[i] for i ∈ 1:length(Ξ.params)))

divide_if_Add(x::Add, divisor) = similarterm(x,operation(x),arguments(x).//divisor)
divide_if_Add(x, divisor) = x//divisor

#arguments_general(x::Union{Add,Mul,Pow}) = arguments(x)
#arguments_general(x::Sym) = [x]
seperate_orders(x::Sym,ps::Union{APT,Vector{DataType},DataType}) = Dict([Tuple(pert(x,ps)) => x])
function seperate_orders(x::Union{Add,Mul,Pow}, ps::Union{APT,Vector{DataType},DataType}; divide=false) 
	new = expand(x)
	# Need a general dictionary that can hold lots of different types
	if divide
		thedicts = [Dict{Tuple{Number,Number},Union{Add,Mul,Pow,Sym,Term}}([Tuple(pert(a,ps)) => divide_if_Add(a,pert_like(pert(a,ps),ps))]) for a ∈ arguments(new)]
	else
		thedicts = [Dict{Tuple{Number,Number},Union{Add,Mul,Pow,Sym,Term}}([Tuple(pert(a,ps)) => a]) for a ∈ arguments(new)]
	end
	final_dict = merge(operation(x), thedicts...)
	final_dict
end
seperate_orders(ps::Union{APT,Vector{DataType},DataType}; divide=false) = (x->seperate_orders(x,ps;divide=divide))


function seperate_all_orders(x::Union{Add,Mul,Pow}, ps::Union{APT,Vector{DataType},DataType})
	
end
	
md"> Adding Perturbations"
end

# ╔═╡ 6a6f7542-acfd-4f48-bff5-26ea3bfa6eb6
begin

abstract type AbstractAnalyticOperator <: AbstractSlotStructure end

# at the moment assume complete analyticity over all non perturbative slots
struct AnalyticOperator <: AbstractAnalyticOperator 
	indices::Vector{Int64}
end

struct OperatorExpansion <: AbstractSlotStructure
	order::Vector{Int64}
end

is_expansion(x) = false
is_expansion(x::Slots) = Type{OperatorExpansion} ∈ keys(x.slot_structures)
is_expansion(x::Term{NewTensor}) = is_expansion(Slots(x))
	
expansion_order(x) = (nothing,)
expansion_order(x::Slots) = x.slot_structures[Type{OperatorExpansion}].order
expansion_order(x::Term{NewTensor}) = expansion_order(Slots(x))

order_match(x, order::Union{Tuple,Vector}) = all(expansion_order(x) .== order)

is_analytic(x::Slots) = Type{AnalyticOperator} ∈ keys(x.slot_structures)
is_analytic(x::Term{NewTensor}) = is_analytic(Slots(x))
analytic_indices(x::Slots) = x.slot_structures[Type{AnalyticOperator}].indices

is_an_Add(x::Add) = true
is_an_Add(x) = false
should_expand_analytic(x) = false
function should_expand_analytic(x::Term{NewTensor})
	if !is_analytic(x)
		return false
	end
	any(is_an_Add.(arguments(x)))
end

fixed_order_term(a::Vector,b::Union{Vector{Int},Tuple}) = Iterators.flatten([collect(repeat([a[i]],b[i])) for i ∈ 1:length(b)]) |> collect

function all_combs(target_order, n_slots)
	a = Iterators.product(repeat([0:target_order],n_slots)...) |> collect
	[collect(i) for i ∈ vcat(a...) if sum(i) ≤ target_order]
end

is_zero(x::Number) = (x == 0)
is_zero(x::Union{Sym,Add,Mul,Term,Pow}) = false

function is_zero_at_this_order(args::Vector, order::Vector{Int})
	inds = findall(x -> x != 0, order)
	any(is_zero.(args[inds]))
end

function order_to_symmetry(order)
	n = 1:sum(order)
	y = cumsum(order)
	[collect((i-1 == 0 ? 1 : y[i-1]+1):y[i]) for i ∈ 1:(length(y))]
end

function make_expanded_term_like(old::Term{NewTensor}, args::Vector, order::Vector{Int}; Linearity=Multilinear, Symmetry=TotalSymmetry, OperatorExpansion=OperatorExpansion)
	F = operation(old)
	display = TensorDisplay(old)
	new_display = append_sup(display, "$(order)")
	old_slot_struct = Slots(old)
	new_slot_number = sum(order)
	NewSlotStructureNeeded = Slots{new_slot_number}(
		Multilinear(collect(1:new_slot_number)),
		TotalSymmetry(order_to_symmetry(order)),
		OperatorExpansion(order)
	)
	NewSlotStructureNeeded = merge_together(old_slot_struct, NewSlotStructureNeeded)
	remove_structure!(NewSlotStructureNeeded, Type{AnalyticOperator})
	new_metadata = Base.ImmutableDict(Type{Slots} => NewSlotStructureNeeded)
	new_metadata = merge(F.metadata, new_metadata)
	F_new = operator(new_display, NewSlotStructureNeeded, new_metadata)
	F_new(fixed_order_term(args,order)...)
end

order_seperate(x::Add, Ξ::APT) = seperate_orders(x,Ξ)
order_seperate(x::Union{Mul,Term,Sym,Pow}, Ξ::APT) = Dict([Tuple(pert(x,Ξ)) => x])

function seperate_zeroth_and_higher(x::Union{Add,Mul,Term,Sym,Pow},Ξ::APT)
	ords = order_seperate(x,Ξ)
	zeroth = get(ords,(0,0),0)
	non_zeroth = x - zeroth
	(zeroth,non_zeroth)
end
seperate_zeroth_and_higher(Ξ::APT) = (x -> seperate_zeroth_and_higher(x,Ξ))
	
function analytic_expand_term(x::Term{NewTensor}, total_order::Int, Ξ::APT)	
	N_slots = number_of_slots(x)
	F = operation(x)
	args = arguments(x)
	terms = seperate_zeroth_and_higher(Ξ).(expand.(arguments(x)))
	zeroth_args = [a[1] for a in terms]
	new_args = [a[2] for a in terms]
	all_extra_terms = [make_expanded_term_like(x, new_args, order) for order ∈ all_combs(total_order, N_slots) if (sum(order) != 0) && !(is_zero_at_this_order(new_args,order))]
	if length(all_extra_terms) == 0
		linear_terms = 0
	else
		linear_terms = sum(all_extra_terms)
	end
	return F(zeroth_args...) + linear_terms
end
analytic_expand_term(Ξ::APT, total_order::Int) = (x -> analytic_expand_term(x, total_order, Ξ))


r2(Ξ::APT, total_order::Int) = @rule ~x::(z -> should_expand_analytic(z)) => analytic_expand_term(x,total_order,Ξ)

expand_analytic(x, Ξ::APT, total_order::Int) = simplify(x,Prewalk(PassThrough(r2(Ξ, total_order))))

expand_analytic(Ξ::APT, total_order::Int) = (x -> expand_analytic(x, Ξ, total_order))
	
md"> Perturbation Expansions implemented" 
end

# ╔═╡ f7344d42-c926-485b-a041-11c980cd3522
begin
using BenchmarkTools

### Define Perturbations
ϵ = variable("\\epsilon", Type{:Background} => 1)
η = variable("\\eta", Type{:Wave} => 1)

Ξ = PerturbationParameters(Dict([Type{:Background} => ϵ, Type{:Wave} => η]))

### Define expansion variables
n_expansion = 2
g = [variable("g_{$(i)}", Dict([Type{NotScalar{Multilinear}} => true])) for i ∈ 0:n_expansion]
#h = [variable("h_{$(i)}", Dict([Type{NotScalar{Multilinear}} => true])) for i ∈ 0:n_expansion]
h = variable("h", Dict([Type{NotScalar{Multilinear}} => true]))
θ = [variable("\\theta_{$(i)}", Dict([Type{NotScalar{Multilinear}} => true])) for i ∈ 0:n_expansion]
#ϕ = [variable("\\phi_{$(i)}", Dict([Type{NotScalar{Multilinear}} => true])) for i ∈ 0:n_expansion]
ϕ = variable("\\phi", Dict([Type{NotScalar{Multilinear}} => true]))

### Define operators
G = operator(TensorDisplay("G","ab",""), Slots{1}(AnalyticOperator([1])))
T = operator(TensorDisplay("T","ab","\\vartheta"), Slots{2}(AnalyticOperator([1,2])))
V = operator(TensorDisplay("V","ab","\\textrm{int}"), Slots{2}(AnalyticOperator([1,2])))
W = operator(TensorDisplay("\\mathcal{W}","A",""), Slots{2}(AnalyticOperator([1,2])))
ρ = operator(TensorDisplay("R","A",""), Slots{2}(AnalyticOperator([1,2])))

ϵ_list = [ϵ^i for i ∈ 0:2]
#args = [sum(θ.*ϵ_list) + η*sum(ϕ.*ϵ_list),sum(g.*ϵ_list) + η*sum(h.*ϵ_list)]
#args = [sum(θ.*ϵ_list) + η*ϕ,sum(g.*ϵ_list) + η*h]
args = [ϵ*θ[2] + ϵ^2*θ[3] + η*ϕ, g[1] + ϵ^2*g[3] + η*h]

expr = G(args[2]) - T(args...) - ϵ*V(args...)
expr2 = W(args...) - ϵ*ρ(args...)
	
end

# ╔═╡ faa4e0e7-0277-403c-9b3f-cad7580773db
begin

r3(p::Function) = @rule ~x::(z -> (is_expansion(z) && p(z))) => 0	
set_orders_to_zero(x, p::Function) = simplify(x,Prewalk(PassThrough(r3(p))))
set_orders_to_zero(p::Function) = (x -> set_orders_to_zero(x, p))

function terms_zero(x::Term{NewTensor})
	order = expansion_order(x)
	result = false
	# Assumption 1: T j ≤ 2
	result |= (TensorDisplay(x).name == "T") && (order[1] < 2)
	# Assumption 1: W j = 0
	result |= (TensorDisplay(x).name == "\\mathcal{W}") && (order[1] == 0)
	# Assumption 1: V j = 0
	result |= (TensorDisplay(x).name == "V") && (order[1] == 0)
end


md"""
> Putting in assumptions based on order
> - Assumption 1:
>  $T^{\vartheta[j,k]}[...] = 0\textrm{ for }j \leq 2$
> 
>  $W^{[j,k]}[...] = 0\textrm{ for }j = 0$
>
>  $V^{[j,k]}_{\textrm{int}}[...] = 0\textrm{ for }j = 0$
"""
end

# ╔═╡ 9527564e-2d38-4e35-b27a-5576e957ba7f
gg = ((expr |> expand_analytic(Ξ,4) |> expand_linear(Multilinear) |> canonicalize ))|> set_orders_to_zero(terms_zero)

# ╔═╡ af8a86fa-222c-49f8-95b6-e9d0843e9ddd
(expr2 |> expand_analytic(Ξ,4) |> expand_linear(Multilinear) |> canonicalize |> set_orders_to_zero(terms_zero) |> seperate_orders(Ξ,divide=true))[(1,2)] |> (x -> arguments(arguments(x)[1])[2] ) |> TensorDisplay

# ╔═╡ 2271cb49-48bf-4d21-991e-8dd69c39d896
begin
r4(target_order, Ξ::APT) = @rule ~x::(z -> !all(pert(z,Ξ) .≤ target_order)) => 0
keep_lower(x, target_order, Ξ::APT) = simplify(expand(x),Prewalk(PassThrough(r4(target_order,Ξ))))
keep_lower(target_order, Ξ::APT) = (x -> keep_lower(x, target_order, Ξ))

gg |> keep_lower((2,1),Ξ) |> seperate_orders(Ξ,divide=true)
end

# ╔═╡ 5a482ea3-9333-463d-b4e8-28936a3116df
expr2 |> expand_analytic(Ξ,4) |> expand_linear(Multilinear) |> keep_lower((1,1),Ξ) |> canonicalize |> set_orders_to_zero(terms_zero)

# ╔═╡ a5eee5a1-2edc-43c7-a24b-59ed106b8c97
bb = ((expr2 |> expand_analytic(Ξ,4) |> expand_linear(Multilinear) |> canonicalize ))|> set_orders_to_zero(terms_zero)|> keep_lower((1,1),Ξ) |> seperate_orders(Ξ,divide=true)

# ╔═╡ 607b388a-47d2-423d-a1a7-f2ad4943200b
(arguments(arguments(gg)[1])[2]) |> terms_zero

# ╔═╡ 2e58a055-39d9-4efd-a5f4-ff339993ccda
md"""
#### Slots
There are `operators`with `Slots`, where the slots can have many `SlotStructures`, they can be:
- `Multilinear` under a particular subset of Slots or/and
- `Symmetric` under a particular subset of Slots
The methods that help use these simplifications are:

|SlotStructure| Method|
|-------------|--------------|
|`Multilinear`|`expand_linear`|
|`Symmetric`|`canonicalize`|
|`AnalyticOperator`| `expand_analytic`|

> How does one `expand_linear`?
>
> - Well you look at a candidate term, and see all it's arguments, and find which slot they belong to.
> - Make a list of arguments in linear slots. First run the `expand` function on the terms in the argument list.
>   - If the argument is an `Add` run `arguments(term)` to get a list of arguments
>   - If the argument is anything else, just encapsulate it in a list.
> - Now loop over everything in this list of lists.
>   - If the argument is a `Mul` seperate it into scalars and non-scalars. There should be a function `is_scalar(x::CandidateTerm, y::Type{<:Multilinear})` defined for this particular subtype of `Multilinear`, which can be used to take a `Mul` and seperate it into scalars and non-scalars.
>   - `scalar_seperate(x) = (1,x)` while `scalar_seperate(x::Mul) = do the calculation...`
> - Now you have a list of list of tuples. Loop over the outer product the list of lists and use that to construct a term like `prod(x[1])*similarterm(term, operation(term), arguments(x[2]))` (something like that...)

This makes sense

> How does one `canonicalize`?
>
> - Well you look at a candidate term, see all it's arguments, and find which slots belong to a particular symmetry structure.
> - Actually I think one would just have to do `ButlerPortugalCanonicalization`

Right now we actually want to only support `TotallySymmetric` slot structures. 
In this, it's particularly straightforward. 
> How does one `canonicalize` a `TotallySymmetric` slot structure?
> - Define a lexical ordering `≤ₒ` function that acts on symbolic types.
> - Go to all arguments of a candidate term, and if thety have a TotallySymmetric slot structure, then order them based on the lexical ordering function.
> - That's it.
"""

# ╔═╡ 6859d3fc-1499-449d-b181-ddd970b02120
begin
md"""
#### Perturbations
We can also define perturbation variables and expand perturbations in multiple scalars. These perturbation variables aren't really very different from normal variables but they carry something like a tag saying I am a *something* type of perturbation and I am of order *n*. 

Whenever you ask for the expansion order of any particular expression, you also need to provide a list of perturbation types you care about. These are held in a type of object called `PerturbationParameters` which just hold perturbation parameters that are deemed of current use, and this parameter is passed around whenever we want to do any calculation which is an expansion or some sort of simplification. 

An important method is `pert(x, Ξ::PerturbationParameter)` which will take an expression and attempt to tell you the order of the expression with respect to each of the perturbation parameters in Ξ. 
"""
end

# ╔═╡ 83a84e8e-45b5-4189-b95c-06764b5635cc
begin
md"""
#### Operator Expansions
We can also take general expressions of non-linear operators that depend on perturbations, and then expand them to multilinear operators that are totally symmetric in each particular *type* of slot. So for example:

```math
F(g+\epsilon h,\theta + \eta\phi) = F + \eta F^{[0,1]}[\phi] + \epsilon F^{[1,0]}[h] + \eta\epsilon F^{[1,1]}[h,\phi] + ...
```


"""
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
SymbolicUtils = "d1185830-fcd6-423d-90d6-eec64667417b"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[compat]
BenchmarkTools = "~1.3.1"
LaTeXStrings = "~1.3.0"
SymbolicUtils = "~0.19.7"
Symbolics = "~4.4.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractAlgebra]]
deps = ["GroupsCore", "InteractiveUtils", "LinearAlgebra", "Markdown", "Random", "RandomExtensions", "SparseArrays", "Test"]
git-tree-sha1 = "b859af958bc9440b44e6d3013fe5a34b18d8a7fc"
uuid = "c3fe647b-3220-5bb0-a1ea-a7954cac585d"
version = "0.23.0"

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "81f0cb60dc994ca17f68d9fb7c942a5ae70d9ee4"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "5.0.8"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AutoHashEquals]]
git-tree-sha1 = "45bb6705d93be619b81451bb2006b7ee5d4e4453"
uuid = "15f4f7f2-30c1-5605-9d31-71845cf9641f"
version = "0.2.0"

[[BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "b15a6bc52594f5e4a3b825858d1089618871bf9d"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.36"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

[[Bijections]]
git-tree-sha1 = "705e7822597b432ebe152baa844b49f8026df090"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.1.3"

[[BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "28bbdbf0354959db89358d1d79d421ff31ef0b5e"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.3"

[[CPUSummary]]
deps = ["CpuId", "IfElse", "Static"]
git-tree-sha1 = "baaac45b4462b3b0be16726f38b789bf330fcb7a"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.1.21"

[[Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[CloseOpenIntervals]]
deps = ["ArrayInterface", "Static"]
git-tree-sha1 = "f576084239e6bdf801007c80e27e2cc2cd963fe0"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.6"

[[Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "b153278a25dd42c65abbf4e62344f9d22e59191b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.43.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[CompositeTypes]]
git-tree-sha1 = "d5b014b216dc891e81fea299638e4c10c657b582"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.2"

[[CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "32d125af0fb8ec3f8935896122c5e345709909e5"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.0"

[[DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "28d605d9a0ac17118fe2c5e9ce0fbb76c3ceb120"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.0"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "f206814c860c2a909d2a467af0484d08edd05ee7"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.57"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "StaticArrays", "Statistics"]
git-tree-sha1 = "5f5f0b750ac576bcf2ab1d7782959894b304923e"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.5.9"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[DynamicPolynomials]]
deps = ["DataStructures", "Future", "LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Pkg", "Reexport", "Test"]
git-tree-sha1 = "d0fa82f39c2a5cdb3ee385ad52bc05c42cb4b9f0"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.4.5"

[[EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "d064b0340db45d48893e7604ec95e7a2dc9da904"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.5.0"

[[ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "7a380de46b0a1db85c59ebbce5788412a39e4cb7"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.28"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[Groebner]]
deps = ["AbstractAlgebra", "Combinatorics", "Logging", "LoopVectorization", "MultivariatePolynomials", "MutableArithmetics", "Primes", "Random", "SortingAlgorithms", "StaticArrays"]
git-tree-sha1 = "12e65df23e3f562cfde6811f5b384430b0fa0b27"
uuid = "0b43b601-686d-58a3-8a1c-6623616c7cd4"
version = "0.1.1"

[[GroupsCore]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9e1a5e9f3b81ad6a5c613d181664a0efc6fe6dd7"
uuid = "d5909c97-4eac-4ecc-a3dc-fdd0858a4120"
version = "0.4.0"

[[HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "18be5268cf415b5e27f34980ed25a7d34261aa83"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.7"

[[Hwloc]]
deps = ["Hwloc_jll"]
git-tree-sha1 = "92d99146066c5c6888d5a3abc871e6a214388b91"
uuid = "0e44f5e4-bd66-52a0-8798-143a42290a1d"
version = "2.0.0"

[[Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "303d70c961317c4c20fafaf5dbe0e6d610c38542"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.7.1+0"

[[HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "65e4589030ef3c44d3b90bdc5aac462b4bb05567"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.8"

[[IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[IntegerMathUtils]]
git-tree-sha1 = "f366daebdfb079fd1fe4e3d560f99a0c892e15bc"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "bcf640979ee55b652f3b01650444eb7bbe3ea837"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.4"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[LabelledArrays]]
deps = ["ArrayInterface", "ChainRulesCore", "LinearAlgebra", "MacroTools", "StaticArrays"]
git-tree-sha1 = "fbd884a02f8bf98fd90c53c1c9d2b21f9f30f42a"
uuid = "2ee39098-c373-598a-b85f-a56591580800"
version = "1.8.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "46a39b9c58749eefb5f2dc1178cb8fab5332b1ab"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.15"

[[LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static"]
git-tree-sha1 = "b651f573812d6c36c22c944dd66ef3ab2283dfa1"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.6"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "76c987446e8d555677f064aaac1145c4c17662f8"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.14"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "ChainRulesCore", "CloseOpenIntervals", "DocStringExtensions", "ForwardDiff", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "SIMDDualNumbers", "SLEEFPirates", "SpecialFunctions", "Static", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "4acc35e95bf18de5e9562d27735bef0950f2ed74"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.108"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Metatheory]]
deps = ["AutoHashEquals", "DataStructures", "Dates", "DocStringExtensions", "Parameters", "Reexport", "TermInterface", "ThreadsX", "TimerOutputs"]
git-tree-sha1 = "0886d229caaa09e9f56bcf1991470bd49758a69f"
uuid = "e9d8d322-4543-424a-9be4-0cc815abe26c"
version = "1.3.3"

[[MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "6bb7786e4f24d44b4e29df03c69add1b63d88f01"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.2"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MultivariatePolynomials]]
deps = ["ChainRulesCore", "DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "393fc4d82a73c6fe0e2963dd7c882b09257be537"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.4.6"

[[MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "842b5ccd156e432f369b204bb704fd4020e383ac"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "0.3.3"

[[NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "3114946c67ef9925204cc024a73c9e679cebe0d7"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.8"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "7e597df97e46ffb1c8adbaddfa56908a7a20194b"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.1.5"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "747f4261ebe38a2bc6abf0850ea8c6d9027ccd07"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RandomExtensions]]
deps = ["Random", "SparseArrays"]
git-tree-sha1 = "062986376ce6d394b23d5d90f01d81426113a3c9"
uuid = "fb686558-2515-59ef-acaa-46db3789a887"
version = "0.4.3"

[[RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "ChainRulesCore", "DocStringExtensions", "FillArrays", "LinearAlgebra", "RecipesBase", "Requires", "StaticArrays", "Statistics", "ZygoteRules"]
git-tree-sha1 = "bfe14f127f3e7def02a6c2b1940b39d0dabaa3ef"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.26.3"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Referenceables]]
deps = ["Adapt"]
git-tree-sha1 = "e681d3bfa49cd46c3c161505caddf20f0e62aaa9"
uuid = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"
version = "0.1.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "cdc1e4278e91a6ad530770ebb327f9ed83cf10c4"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.3"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[SIMDDualNumbers]]
deps = ["ForwardDiff", "IfElse", "SLEEFPirates", "VectorizationBase"]
git-tree-sha1 = "62c2da6eb66de8bb88081d20528647140d4daa0e"
uuid = "3cdde19b-5bb0-4aaf-8931-af3e248e098b"
version = "0.1.0"

[[SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "ac399b5b163b9140f9c310dfe9e9aaa225617ff6"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.32"

[[SciMLBase]]
deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "RecipesBase", "RecursiveArrayTools", "StaticArrays", "Statistics", "Tables", "TreeViews"]
git-tree-sha1 = "194a569a247b8180e7171f7ee59dabfd5a095f9f"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.31.3"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "38d88503f695eb0301479bc9b0d4320b378bafe5"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.2"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5ba658aeecaaf96923dce0da9e703bd1fe7666f9"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.4"

[[SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "5309da1cdef03e95b73cd3251ac3a39f887da53e"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.6.4"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "cd56bf18ed715e8b09f06ef8c6b781e6cdc49911"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.4"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c82aaa13b44ea00134f8c9c89819477bd3986ecd"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.3.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "ca9f8a0c9f2e41431dc5b7697058a3f8f8b89498"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.0"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[SymbolicUtils]]
deps = ["AbstractTrees", "Bijections", "ChainRulesCore", "Combinatorics", "ConstructionBase", "DataStructures", "DocStringExtensions", "DynamicPolynomials", "IfElse", "LabelledArrays", "LinearAlgebra", "Metatheory", "MultivariatePolynomials", "NaNMath", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "TermInterface", "TimerOutputs"]
git-tree-sha1 = "bfa211c9543f8c062143f2a48e5bcbb226fd790b"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "0.19.7"

[[Symbolics]]
deps = ["ArrayInterface", "ConstructionBase", "DataStructures", "DiffRules", "Distributions", "DocStringExtensions", "DomainSets", "Groebner", "IfElse", "Latexify", "Libdl", "LinearAlgebra", "MacroTools", "Metatheory", "NaNMath", "RecipesBase", "Reexport", "Requires", "RuntimeGeneratedFunctions", "SciMLBase", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicUtils", "TermInterface", "TreeViews"]
git-tree-sha1 = "aab7c217abd7427e91004a2486bef9af42a1047a"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "4.4.3"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TermInterface]]
git-tree-sha1 = "7aa601f12708243987b88d1b453541a75e3d8c7a"
uuid = "8ea1fca8-c5ef-4a55-8b96-4e9afe9c9a3c"
version = "0.2.3"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "f8629df51cab659d70d2e5618a430b4d3f37f2c3"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.0"

[[ThreadsX]]
deps = ["ArgCheck", "BangBang", "ConstructionBase", "InitialValues", "MicroCollections", "Referenceables", "Setfield", "SplittablesBase", "Transducers"]
git-tree-sha1 = "d223de97c948636a4f34d1f84d92fd7602dc555b"
uuid = "ac1d9e8a-700a-412c-b207-f0111f4b6c0d"
version = "0.1.10"

[[TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "d3bfa83f95c706485de9ae755a23a6ce5b1c30a3"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.18"

[[Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "c76399a3bbe6f5a88faa33c8f8a65aa631d95013"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.73"

[[TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "Hwloc", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static"]
git-tree-sha1 = "ff34c2f1d80ccb4f359df43ed65d6f90cb70b323"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.31"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─4c467392-ccde-11ec-15c0-4f532000315f
# ╟─9bd50b69-1f6d-4a5a-855e-15d70657742f
# ╠═977b2170-886d-40d8-ada2-29385c8c8bde
# ╠═eb6f19bd-f750-478b-967e-39710e4f42c2
# ╟─004ab9ee-4562-421a-ad7c-eaa91c8f96db
# ╟─1f229cd6-8ab9-4c37-a598-b9a3b2d65c94
# ╠═893d798e-7376-43bc-99b1-2e65f45f1c18
# ╠═223ae6fa-bbae-42ba-a4ea-6dc2814aa521
# ╠═6a6f7542-acfd-4f48-bff5-26ea3bfa6eb6
# ╠═faa4e0e7-0277-403c-9b3f-cad7580773db
# ╠═f7344d42-c926-485b-a041-11c980cd3522
# ╠═9527564e-2d38-4e35-b27a-5576e957ba7f
# ╠═af8a86fa-222c-49f8-95b6-e9d0843e9ddd
# ╠═5a482ea3-9333-463d-b4e8-28936a3116df
# ╠═a5eee5a1-2edc-43c7-a24b-59ed106b8c97
# ╠═2271cb49-48bf-4d21-991e-8dd69c39d896
# ╠═607b388a-47d2-423d-a1a7-f2ad4943200b
# ╟─2e58a055-39d9-4efd-a5f4-ff339993ccda
# ╟─6859d3fc-1499-449d-b181-ddd970b02120
# ╟─83a84e8e-45b5-4189-b95c-06764b5635cc
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
